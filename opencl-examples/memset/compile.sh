#!/bin/sh

clang++ -emit-llvm main.cpp -c -o main.bc
opt -o main.pass.bc -load ~/583Project/build/mypass/LLVMPJT.so -hello < main.bc > /dev/null
clang main.pass.bc -o main -lOpenCL -L/opt/amdgpu-pro/lib/x86_64-linux-gnu/
