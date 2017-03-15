#!/usr/bin/env bash

if [ "$1" == 'clean' ]
then
	rm -rf machinelearning cmake_install.cmake CMakeCache.txt Makefile CMakeFiles/
	exit
fi

if [ ! -f Makefile ]
then
	cmake .
fi

make
./machinelearning

