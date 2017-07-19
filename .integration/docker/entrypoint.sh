#!/bin/bash
set -e
if [ -n "$1" && -n "$2" ]; then
    curl -sSL $1 | $2
else
    echo first  parameter is the url where the config file can be found.
    echo second parameter is the executable program to execute.
fi
