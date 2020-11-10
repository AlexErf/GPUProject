# Memset example
This GPU kernel `main.cl` simply has each thread write its ID to memory. So, memory looks like:
[1, 2, 3, 4, ...] when it is done.

The OpenCL Driver `main.cpp` uses the OpenCL C interface to load the compiled version of `main.cl` (by default, `kernel.exe`, but you can pass in a command line argument to override that, i.e. `main.exe my_filename_here.exe`) and run it.

It works by
1. getting a "platform"
2. Finding a GPU device on that platform
3. Creating a context associated with that device
4. Creating a command queue to interact with that device
5. Reading the binary bites from the file 
6. Creating a program with the binary for the device
7. Building it
8. Creating the kernel to call
9. Allocating GPU memory for the kernel to write to
10. Setting that as the argument to the kernel
11. Running the kernel (enqueue it to be run on the command queue, then wait for the queue to be flushed)
12. It gets the results out by mapping the GPU memory into the host's virtual address space (One should be able to copy the results from GPU to CPU memory as well).

# To Build
Use the associated `Makefile`. You should be able to use your version of `clang` provided earlier. It also appears to work with the system default `clang` as well, which is cool. You should also be able to use ROCm's script to independently compile OpenCL kernels (found in `/opt/rocm/bin/clang-ocl`, usage is like: `clang-ocl main.cl -o kernel.exe`).

## Compiling the host code
`make main.exe` will do the trick.

## Compiling the GPU code
`make kernel.exe` should *just work*. If you want to see how ROCm would compile it, you can use `make kernel_rocm.exe`. This will invoke the `clang-ocl` wrapper mentioned previously. I have modified the wrapper to print all of the steps it does in building it, so you could even write the output to a bash script & tweak it from there!

# Running
`./main.exe`. This should print numbers 0 to 512.
