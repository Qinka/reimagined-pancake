#!/bin/bash

set -e # error -> exit(1)

if [ -n "$IS_DOCKER" ] && [ -n "$TARGET" ] && [ -n "$EXECUTABLE" ]; then
    echo build and update the docker image
    echo
    echo setup target NAME TAG
    export IMAGE=qinka/reimagined-pancake
    export TAG=$TARGET$TARGET_NAME-backend
    echo
    echo copy files
    cd $TRAVIS_BUILD_DIR
    echo create temporary directory
    mkdir -p docker.tmp/bin
    echo coping...
    sudo cp $HOME/.local/bin/$EXECUTABLE docker.tmp/bin
    if [ x"$TARGET_NAME" = x"-knn-llvm-native" ]; then
	export DOCKERFILE=Dockerfile.llvm
    elif [ x"$TARGET_NAME" = x"-knn-llvm-ptx" ]; then
	export DOCKERFILE=Dockerfile.cuda
    elif [ x"$TARGET_NAME" = x"-nn" ]; then
	export DOCKERFILE=Dockerfile.tf
	export TAG=$TAG-$TF_TYPE
	cp -r $TRAVIS_BUILD_DIR/../libtensorflow docker.tmp/
    else
	export DOCKERFILE=Dockerfile
    fi
    sudo cp $TRAVIS_BUILD_DIR/.integration/docker/$DOCKERFILE docker.tmp/Dockerfile
    sudo cp $TRAVIS_BUILD_DIR/.integration/docker/entrypoint.sh docker.tmp
    echo build docker image
    cd docker.tmp
    docker build -t $IMAGE:$TAG .
    docker push $IMAGE
fi

    
