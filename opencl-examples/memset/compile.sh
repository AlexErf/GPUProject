#!/bin/sh

clang++ -emit-llvm main.cpp -c -o main.bc
opt -o main.pass.bc -load ~/GPUProject/build/mypass/LLVMPJT.so -hello < main.bc > /dev/null
clang main.pass.bc -o main -lOpenCL -L/opt/rocm/opencl/lib/
