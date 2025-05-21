#!/bin/bash -l

cmake -S . -B build 
cmake --build build -j
