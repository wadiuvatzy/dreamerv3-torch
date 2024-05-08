for i in {22..99}; do
    if ! [ -e /tmp/.X${i}-lock ]; then
        DISPLAY_NUM=$i
        break
    fi
done

# Start Xvfb on the found display number
echo "Starting Xvfb on display ${DISPLAY_NUM}"
Xvfb :${DISPLAY_NUM} -screen 0 1024x768x24 &

# Export the DISPLAY variable for your jobs
export DISPLAY=:${DISPLAY_NUM}.0