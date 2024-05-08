import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd
import re
from copy import deepcopy, copy
import theseus as th


to_np = lambda x: x.detach().cpu().numpy()

def linear_schedule(schdl, step):
	"""
	Outputs values following a linear decay schedule.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	try:
		return float(schdl)
	except ValueError:
		match = re.match(r'linear\((.+),(.+),(.+),(.+)\)', schdl)
		if match:
			init, final, start, duration = [float(g) for g in match.groups()]
			mix = np.clip((step - start) / duration, 0.0, 1.0)
			return (1.0 - mix) * init + mix * final
	raise NotImplementedError(schdl)

class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

        self.a_loss_optim = torch.optim.Adam(self.parameters(), lr=config.aug_lr)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        # policy_output, state = self._policy(obs, state, training)
        if self._step < self._config.planning_start:
            policy_output, state = self._policy(obs, state, training)
        else:
            print("start planning")
            policy_output, state = self._policy_with_plan(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state
    
    def _get_logprob(self, obs, state, action_input):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        actor = self._task_behavior.actor(feat)
        logprob = actor.log_prob(action_input)
        # import pdb; pdb.set_trace()
        return logprob

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)
            

    def _policy_with_plan(self, obs, state, training=False):
        if state is None:
            latent = action = None
        else:
            latent, action = state

        original_obs = copy(obs)
        original_state = copy(state)
        # maybe add more update of dynamics and encoder and reward 
        # BACK_PLAN mode in theseus
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        cur_latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        feat = self._wm.dynamics.get_feat(cur_latent)

        horizon = int(min(self._config.planning_horizon, linear_schedule(self._config.planning_horizon_schedule, self._step)))

        with torch.no_grad():
            pi_actions = torch.empty(horizon, 4, self._config.num_actions, device=self._config.device)
            current_latent = copy(cur_latent)
            for t in range(horizon):
                feat = self._wm.dynamics.get_feat(current_latent)
                actor = self._task_behavior.actor(feat)
                a = actor.mode()
                pi_actions[t] = a
                # current_latent = self._wm.dynamics.imagine_with_action(a, current_latent)
                current_latent = self._wm.dynamics.img_step(current_latent, a)
        pi_actions = pi_actions.view(1, -1)
        init_actions = copy(pi_actions)

        actions = torch.zeros(horizon, 4, self._config.num_actions, device=self._config.device)
        if hasattr(self, '_prev_actions'):
            actions[:-1] = self._prev_actions[1:]
        actions = actions.view(1, -1)

        actions = th.Vector(tensor=actions, name="actions")
        # import pdb; pdb.set_trace()
        latent_stoch = th.Vector(tensor=cur_latent["stoch"].view(1, -1), name="latent_stoch")
        latent_deter = th.Vector(tensor=cur_latent["deter"].view(1, -1), name="latent_deter")
        latent_logit = th.Vector(tensor=cur_latent["logit"].view(1, -1), name="latent_logit")

        for params in self._wm.parameters():
            params.requires_grad = False

        def value_cost_fn(optim_vars, aux_vars):
            latent_stoch, latent_deter, latent_logit = aux_vars
            latent = {"stoch": latent_stoch.tensor.reshape([4, 32, 32]), "deter": latent_deter.tensor.reshape([4, 512]), "logit": latent_logit.tensor.reshape([4, 32, 32])}
            actions = optim_vars[0].tensor.view(horizon, 4, self._config.num_actions)
            actions = torch.clamp(actions, -1, 1)
            G, discount = 0, 1
            for t in range(horizon):
                feat = self._wm.dynamics.get_feat(latent)
                reward = self._wm.heads["reward"](feat).mode()
                G += discount * reward
                discount *= self._config.discount
                latent = self._wm.dynamics.img_step(latent, actions[t], sample=False)

            feat = self._wm.dynamics.get_feat(latent)
            G += discount * self._task_behavior.value(feat).mode()
            err = -G.nan_to_num_(0) + 2000
            err = torch.sum(err).view(1, -1)
            # print("err", err)
            return err 
        
        optim_vars = [actions]
        aux_vars = [latent_stoch, latent_deter, latent_logit]
        cost_function = th.AutoDiffCostFunction(
			optim_vars, value_cost_fn, 1, aux_vars=aux_vars, name="value_cost_fn", 
			#autograd_mode=th.AutogradMode.LOOP_BATCH
		)
        objective = th.Objective()
        objective.to(device=self._config.device, dtype=torch.float32)
        objective.add(cost_function)
        optimizer = th.LevenbergMarquardt(
			objective,
			th.CholeskyDenseSolver,
            max_iterations=self._config.planning_iters,
            step_size=self._config.planning_step_size,
        )
        theseus_optim = th.TheseusLayer(optimizer)
        theseus_optim.to(device=self._config.device, dtype=torch.float32)
        theseus_inputs = {
            "actions": init_actions,
            "latent_stoch": cur_latent["stoch"].view(1, -1),
            "latent_deter": cur_latent["deter"].view(1, -1),
            "latent_logit": cur_latent["logit"].view(1, -1),
        }  
        updated_inputs, info = theseus_optim.forward(
            theseus_inputs, optimizer_kwargs={"track_best_solution": True, 
        "verbose": False, "damping": self._config.planning_damping, "backward_mode": self._config.planning_backward_mode, "backward_num_iterations": self._config.planning_backward_num_iterations,})

        updated_actions = updated_inputs['actions']
        best_actions = info.best_solution['actions']
        best_actions = best_actions.view(horizon, 4, self._config.num_actions)
        self._prev_actions = best_actions
        # logprob = self._get_logprob(obs, state, best_actions[0].nan_to_num_(0))

        action = best_actions[0].nan_to_num_(0).to(self._config.device)
        action = torch.clamp(action, -1, 1)
        logprob = self._get_logprob(original_obs, original_state, action)
        policy_output = {"action": action, "logprob": logprob}
        state = (cur_latent, action)

        # print("init actions", init_actions)
        # print("best actions", best_actions)

        for params in self._wm.parameters():
            params.requires_grad = True

        if not training:
            return policy_output, state

        # update model
        self.a_loss_optim.zero_grad()
        a_t = copy(action)
        next_latent = self._wm.dynamics.img_step(cur_latent, a_t)
        value = self._task_behavior.value(next_latent).mode()
        for params in self._task_behavior.parameters():
            params.requires_grad = False
        value_cost = -value
        value_cost.backward()
        self.a_loss_optim.step()
        for params in self._task_behavior.parameters():
            params.requires_grad = True


        return policy_output, state

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)
    make = lambda mode, id: make_env(config, mode, id)
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs_with_plan.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
