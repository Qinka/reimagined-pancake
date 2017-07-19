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
