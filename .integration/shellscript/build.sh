#!/bin/bash

set -e # error -> exit(1)

if [ -n "$TARGET" ]; then
    echo output ghc version
    ghc -V
    if [ -n "$STACKSOLVER" ]; then
	echo using stack solver "$TRAVIS_BUILD_DIR/.stack-yaml/$STACKSOLVER"
	export STACKFILE=" --stack-yaml $TRAVIS_BUILD_DIR/.stack-yaml/$STACKSOLVER "
    fi
    if [ -n "$LLVM" ]; then
	echo using LLVM-$LLVM
	export LLVMFLAG=" --ghc-options -fllvm --ghc-options -pgmlo --ghc-options opt-$LLVM --ghc-options -pgmlc --ghc-options llc-$LLVM "
    fi
    echo setup O3 flag
    export O3FLAG=' --ghc-options -O3 '
    echo
    echo run stack install
    stack install $TARGET $TARGET_FLAG $STACKFILE $O3FLAG $LLVMFLAG -v
fi

