HOST=ckarch
USER=chrisk
WHEEL=$(basename -- $(ls wheel/*.whl))

# Build image
echo $USER $HOST
docker build \
    --build-arg USER=$USER \
    --build-arg HOST=$HOST \
    --build-arg WHEEL=$WHEEL \
    -t deepred/lynred .

xhost +local:docker

# Run Image
docker run \
    -it --rm\
    --name face_detection_IR \
    -h $HOST \
    --privileged \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -v $PWD:/workspace \
    deepred/lynred bash
    
   
