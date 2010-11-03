#!/bin/bash
set -ex
topdir=$(pwd)

echo "Build som_pak..."
cd tools/som_pak/ && make -f makefile.unix && cd ${topdir} 

#echo "Please run \"export MRSOM_ROOT=`pwd`\""
#if [ -z "$MRSOM_ROOT" ]; then
    #echo "export MRSOM_ROOT=`pwd`" >> ~/.bash_profile
    #source ~/.bash_profile
#fi

