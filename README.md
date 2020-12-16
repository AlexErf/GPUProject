# GPUProject

Our implementation is located at `llvm-project/llvm/lib/Target/AMDGPU/BBSchedStrategy.cpp` and `llvm-project/llvm/lib/Target/AMDGPU/BBSchedStrategy.h`

Dependencies:
ROCm: `https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#installing-development-packages-for-cross-compilation`

The following lines handle installing the packages mentioned here: `https://rocmdocs.amd.com/en/latest/Installation_Guide/Software-Stack-for-AMD-GPU.html`

`sudo apt update`

`sudo apt install rocm-dev`

To build:

Start in the `llvm-project` directory

`mkdir build && cd build`

``cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=`pwd` -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86"  -DLLVM_ENABLE_EXPENSIVE_CHECKS=ON -DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt" ../llvm``

`ninja clang`

To run (example given will run the demo):

Start in the project root directory

`cd tests`

Add the newly compiled clang to the path: `export PATH=/path/to/build/bin:$PATH`

`which clang` should now yield the newly built clang executable from within the build directory

`make CLI_0068_2F31C061_source.cl_exe`

The output from the clang execution will be directed to the file `demoout.txt`.  This can be modified from within the Makefile.
