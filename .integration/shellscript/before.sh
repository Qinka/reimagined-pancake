#!/bin/bash

set -e # error -> exit(1)

echo
echo The intefration script for this repo
echo
echo Before-install setup
echo
echo Update apt
${SUDO} apt update
echo
echo install wget
${SUDO} apt install -y wget
echo Fetch the system\'s name
export OS_CORENAME=$(lsb_release -a | grep Codename | awk '{print $2}')
echo $OS_CORENAME
export OS_DISTRIBUTOR=$(lsb_release -a | grep Description | awk '{print $2}')
echo $OS_DISTRIBUTOR
echo
if [ -n "$LLVM" ]; then
  echo install llvm
  wget -O - http://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
  echo deb http://apt.llvm.org/$OS_CORENAME/ llvm-toolchain-$OS_CORENAME main | sudo tee -a /etc/apt/sources.list.d/llvm.list
  echo deb-src http://apt.llvm.org/$OS_CORENAME/ llvm-toolchain-$OS_CORENAME main | sudo tee -a /etc/apt/sources.list.d/llvm.list
  echo deb http://apt.llvm.org/$OS_CORENAME/ llvm-toolchain-$OS_CORENAME-$LLVM main | sudo tee -a /etc/apt/sources.list.d/llvm.list
  echo deb-src http://apt.llvm.org/$OS_CORENAME/ llvm-toolchain-$OS_CORENAME-$LLVM main | sudo tee -a /etc/apt/sources.list.d/llvm.list
  echo deb http://apt.llvm.org/$OS_CORENAME/ llvm-toolchain-$OS_CORENAME-4.0 main | sudo tee -a /etc/apt/sources.list.d/llvm.list
  echo deb-src http://apt.llvm.org/$OS_CORENAME/ llvm-toolchain-$OS_CORENAME-4.0 main | sudo tee -a /etc/apt/sources.list.d/llvm.list
  sudo apt update
  sudo apt install -y libllvm-$LLVM-ocaml-dev libllvm$LLVM libllvm$LLVM-dbg lldb-$LLVM llvm-$LLVM llvm-$LLVM-dev llvm-$LLVM-runtime lldb-$LLVM-dev
else
  echo without llvm
fi
echo
echo login docker
docker login -e="$DOCKER_EMAIL" -u="$DOCKER_USERNAME" -p="$DOCKER_PASSWORD"
echo
echo setup ghc
export PATH=/opt/ghc/$GHC_VER/bin:$PATH
echo PATH=$PATH
echo
echo install stack
mkdir -p ~/.local/bin
export PATH=$HOME/.local/bin:$PATH
travis_retry curl -L https://www.stackage.org/stack/linux-x86_64 | tar xz --wildcards --strip-components=1 -C ~/.local/bin '*/stack'
echo setting up ghc
stack config set system-ghc --global true
stack path --programs
true
### END ###

### SP  ###
export TMP_PWD=`pwd`
cd $TRAVIS_BUILD_DIR/svm
git clone https://github.com/Qinka/Simple-SVM.git
cd Simple-SVM
git clone https://github.com/Qinka/bindings-svm.git
cd bindings-svm
git clone https://github.com/Qinka/libsvm.git
cd $TMP_PWD
export TMP_PWD=

if [ x"$TARGET_NAME" = x"-knn-llvm-native" ]; then
    echo
    echo Install knn with llvm depends
    echo
    ${SUDO} apt install -y llvm-4.0
elif [ x"$TARGET_NAME" = x"-knn-llvm-ptx" ]; then
    echo
    echo Install knn with cuda depends
    echo
    ${SUDO} apt install -y llvm-4.0
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
    ${SUDO} dpkg -i cuda-repo-ubuntu1404_8.0.61-1_amd64.deb
    ${SUDO} apt update
    ${SUDO} apt install -y cuda nvidia-cuda-dev nvidia-cuda-toolkit
elif [ x"$TARGET_NAME" = x"-nn" ]; then
    echo
    echo Install neural network with cpu or gpu
    echo
    export INSTALL_DIR=/usr/local
    wget "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-linux-x86_64-1.2.1.tar.gz" -O libtensorflow.tar.gz
    ${SUDO} tar -C $INSTALL_DIR libtensorflow.tar.gz  -xz
    mkdir -p $TRAIVS_BUILD_DIR/../libtensorflow
    ${SUDO} tar -C $TRAIVS_BUILD_DIR/../libtensorflow -xz libtensorflow.tar.gz
    ${SUDO} apt update
    ${SUDO} apt install protobuf-compile
fi

