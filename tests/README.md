`CLI_*source.cl` contain kernels from plaidml where the AMDGPU machine scheduler improved the occupancy of the generic LLVM scheduler.

Currently, `CLI_0068_2F31C061_source.cl` is able to be run. To do so, you first have to make the driver via `make`. Then, you have to compile the kernel via `make CLI_0068_2F31C061_source.cl_exe`. You can run it via `./main.exe CLI_0068_2F31C061_source.cl_exe`. The program initalizes the inputs via a `memset` kernel, where each thread initializes one value of the output based on its thread ID. The repo then executes the kernel. The main computation seems to be some form of `a*b+c` executed in parallel, but hard to say (due to computer generated code).

The output of `CLI_0068_2F31C061_source.cl_exe` is stored in `output.txt`. It prints out the contents of the output buffer, when run against the current AMDGPU scheduler.
