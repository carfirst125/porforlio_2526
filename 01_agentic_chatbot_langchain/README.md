### 01. Build docker

sh build_docker_image.sh

### 02. Delete docker image with name <none> reasonned by dupplicated image name.

sh delete_docker_image_name_none.sh

### 03. Run docker image creating container

sh run_docker_image.sh

Script includes remove existed container and rebuilt new container for image.

### Debug code in container

1- check container deployed : docker ps -a  ==> list of container, Up to x minutes... --> still listenning
2- open container terminal ==> check source code
                            |- rerun code in container --> view log



