default: main.exe kernel.exe
main.exe: main.o
	$(CXX) -g -o $@ $^ -lOpenCL -L/opt/rocm/opencl/lib/

kernel.exe: main.cl
	clang -c -emit-llvm -target amdgcn-amd-amdhsa -x cl -D__AMD__=1 -D__gfx1010__=1 -D__gfx1010=1 -D__OPENCL_VERSION__=120 -D__IMAGE_SUPPORT__=1 -O3 -m64 -cl-kernel-arg-info -nogpulib -cl-std=CL1.2 -mllvm -amdgpu-early-inline-all -Xclang -target-feature -Xclang -code-object-v3 -Xclang -cl-ext=+cl_khr_fp64,+cl_khr_global_int32_base_atomics,+cl_khr_global_int32_extended_atomics,+cl_khr_local_int32_base_atomics,+cl_khr_local_int32_extended_atomics,+cl_khr_int64_base_atomics,+cl_khr_int64_extended_atomics,+cl_khr_3d_image_writes,+cl_khr_byte_addressable_store,+cl_khr_gl_sharing,+cl_amd_media_ops,+cl_amd_media_ops2,+cl_khr_subgroups -include /opt/rocm-3.9.0/llvm/lib/clang/12.0.0/include/opencl-c.h -mcpu=gfx1010 -o kernel.exe.orig.bc main.cl
	llvm-link -f -o kernel.exe.linked.bc kernel.exe.orig.bc /opt/rocm-3.9.0/amdgcn/bitcode/opencl.bc /opt/rocm-3.9.0/amdgcn/bitcode/ocml.bc /opt/rocm-3.9.0/amdgcn/bitcode/ockl.bc /opt/rocm-3.9.0/amdgcn/bitcode/oclc_correctly_rounded_sqrt_off.bc /opt/rocm-3.9.0/amdgcn/bitcode/oclc_daz_opt_on.bc /opt/rocm-3.9.0/amdgcn/bitcode/oclc_finite_only_off.bc /opt/rocm-3.9.0/amdgcn/bitcode/oclc_isa_version_1010.bc /opt/rocm-3.9.0/amdgcn/bitcode/oclc_unsafe_math_off.bc
	# opt -o kernel.exe.pass.bc -load ~/583Project/build/mypass/LLVMPJT.so -hello < kernel.exe.linked.bc > /dev/null
	clang -mllvm -debug-only=machine-scheduler -target amdgcn-amd-amdhsa -O3 -m64 -cl-kernel-arg-info -nogpulib -mllvm -amdgpu-internalize-symbols -mllvm -amdgpu-early-inline-all -Xclang -target-feature -Xclang -code-object-v3 -mcpu=gfx1010 -o kernel.exe kernel.exe.linked.bc > output.txt

kernel_rocm.exe: main.cl
	/opt/rocm/bin/clang-ocl -mcpu=gfx1010 main.cl -o $@


kernel.o: main.cl
	clang++ -Xclang -finclude-default-header -target amd-amdgcn-amdhsa -x cl $< -o $@ -c

%.o: %.cpp
	$(CXX) -o $@ -c $<

clean:
	rm -rf *.exe *.o *.bc
