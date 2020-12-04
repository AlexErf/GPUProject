//
// Modified by Nathan Brown
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

// A minimalist OpenCL program.

#include <CL/cl.h>
#include <stdio.h>

#include <unistd.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <cassert>
#include <array>

#define NWITEMS 512
// A simple memset kernel
const char *source =
"kernel void memset(   global float *dst )             \n"
"{                                                    \n"
"    dst[get_global_id(0)] = get_global_id(0);        \n"
"}                                                    \n";
// get size of a filename
off_t fsize(const char *filename) {
    struct stat st;

    if (stat(filename, &st) == 0)
        return st.st_size;

    return -1;
}

void build_kernel(cl_program& program, cl_device_id& device) {
    std::cout << "building program..." << std::endl;
  auto build_error = clBuildProgram( program, 1, &device, NULL, NULL, NULL );
  if( build_error != CL_SUCCESS) {
      std::cerr << "failed to build" << std::endl;
#define CASE(X) case X: {std::cerr << #X << std::endl; break;}
      switch(build_error) {
          CASE(CL_INVALID_PROGRAM);
          CASE(CL_INVALID_VALUE);
          CASE(CL_INVALID_DEVICE);
          CASE(CL_INVALID_BINARY);
          CASE(CL_INVALID_BUILD_OPTIONS);
          CASE(CL_INVALID_OPERATION);
          CASE(CL_COMPILER_NOT_AVAILABLE);
          CASE(CL_BUILD_PROGRAM_FAILURE);
          CASE(CL_OUT_OF_HOST_MEMORY);
      }
      exit(1);
  }
  cl_build_status build_status;
  auto build_info = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, 0);
  if(build_info != CL_SUCCESS || build_status != CL_BUILD_SUCCESS) {
      switch(build_status) {
          CASE(CL_BUILD_ERROR);
          CASE(CL_BUILD_IN_PROGRESS);
      }
      exit(1);
  }
}

int main(int argc, char ** argv)
{
  // 1. Get a platform.
  cl_platform_id platform;
  clGetPlatformIDs( 1, &platform, NULL );

  // 2. Find a gpu device.
  cl_device_id device;
  clGetDeviceIDs( platform,
                  CL_DEVICE_TYPE_GPU,
                  1,
                  &device, NULL);

  // 3. Create a context and command queue on that device.
  cl_context context = clCreateContext( NULL,
                                        1,
                                        &device,
                                        NULL, NULL, NULL);

  cl_command_queue queue = clCreateCommandQueue( context,
                                                 device,
                                                 0, NULL );

  const char* filename = argc == 1 ? "kernel.exe" : argv[1];
  off_t file_size_s = fsize(filename);
  if(file_size_s == -1) {
      std::cerr << "Couldn't get size of binary" << std::endl;
      exit(1);
  }
  size_t file_size = file_size_s;
  unsigned char* bytes = (unsigned char*) malloc(sizeof(unsigned char) * file_size);
  if(!bytes) {
      std::cerr << "malloc returned nullptr" << std::endl;
      exit(1);
  }
  FILE* file_ptr = fopen(filename, "r");
  if(!file_ptr) {
      perror(NULL);
      exit(1);
  }
  auto read = fread(bytes, file_size, 1, file_ptr);
  if(!read) {
      std::cerr << "fread read " << read << " bytes instead of " << file_size << " bytes" << std::endl;
      if(ferror(file_ptr)) {
        perror("File read failed");
      }
      exit(1);
  }
  // 4. Perform runtime source compilation, and obtain kernel entry point.
  const unsigned char* binary = bytes;
  cl_int binary_status = 0, error_status = 0;
  //while(1) {
  cl_program program = clCreateProgramWithBinary( context,
                                                    1,
                                                    &device,
                                                    &file_size,
                                                    &binary,
                                                    &binary_status,
                                                    &error_status);
  if(error_status != CL_SUCCESS) {
      std::cerr << "failed to create program" << std::endl;
      exit(1);
  }


  cl_program memset_prog = clCreateProgramWithSource( context,
                                                  1,
                                                  &source,
                                                  NULL, NULL );

  build_kernel(program, device);
  build_kernel(memset_prog, device);

  std::cout << "Executing kernel" << std::endl;
  cl_int err_code;
  cl_kernel memset_kernel = clCreateKernel( memset_prog, "memset", &err_code );
  if(err_code != CL_SUCCESS) {
      switch(err_code) {
          CASE(CL_INVALID_PROGRAM);
          CASE(CL_INVALID_PROGRAM_EXECUTABLE);
          CASE(CL_INVALID_KERNEL_NAME);
          CASE(CL_INVALID_VALUE);
          CASE(CL_OUT_OF_HOST_MEMORY);
      }
      return 1;
  }
  cl_kernel kernel = clCreateKernel( program, "kernel_c25_sdk_48", NULL );

  // 5. Create a data buffer.
  constexpr size_t min_in1_index = 7424; // inclusive lower bound
  constexpr size_t max_in1_index = 208128; // exclusive upper bound
  cl_mem in1 = clCreateBuffer( context,
                                  CL_MEM_READ_WRITE,
                                   (max_in1_index - min_in1_index) * sizeof(cl_float),
                                  NULL, NULL );
  constexpr size_t min_in2_index = 0; // inclusive lower bound
  constexpr size_t max_in2_index = 2304; // exclusive upper bound
  cl_mem in2 = clCreateBuffer( context,
                                  CL_MEM_READ_WRITE,
                                   (max_in2_index - min_in2_index) * sizeof(cl_float),
                                  NULL, NULL );
  constexpr size_t min_x_t199_index = 0; // inclusive lower bound
  constexpr size_t max_x_t199_index = 200705; // exclusive upper bound
  cl_mem x_t199 = clCreateBuffer( context,
                                  CL_MEM_WRITE_ONLY,
                                   (max_x_t199_index - min_x_t199_index) * sizeof(cl_float),
                                  NULL, NULL );

  std::array<float, max_in1_index - min_in1_index> in1_before;
  cl_float *ptr;
  ptr = (cl_float *) clEnqueueMapBuffer( queue,
                                        in1,
                                        CL_TRUE,
                                        CL_MAP_READ,
                                        0,
                                        (max_in1_index - min_in1_index) * sizeof(cl_float),
                                        0, NULL, NULL, NULL );
  for(size_t i = 0; i < in1_before.size(); ++i) {
      in1_before[i] = ptr[i];
  }

  // Initialize the input buffers via the memset kernel
  clSetKernelArg(memset_kernel, 0, sizeof(in1), (void*) &in1);
  size_t memset_global_work_size = max_in1_index - min_in1_index;
  cl_int retval;
  retval = clEnqueueNDRangeKernel(queue, memset_kernel, 1, NULL, &memset_global_work_size, NULL, 0, NULL, NULL);
  if(retval != CL_SUCCESS) {
      switch(retval) {
      CASE(CL_INVALID_PROGRAM_EXECUTABLE);
      CASE(CL_INVALID_COMMAND_QUEUE);
      CASE(CL_INVALID_KERNEL);
      CASE(CL_INVALID_CONTEXT);
      CASE(CL_INVALID_KERNEL_ARGS);
      CASE(CL_INVALID_WORK_DIMENSION);
      CASE(CL_INVALID_WORK_GROUP_SIZE);
      CASE(CL_INVALID_WORK_ITEM_SIZE);
      CASE(CL_INVALID_GLOBAL_OFFSET);
      }
      assert(false &&  "could not enqueue memset kernel the first time");
  }
  clFlush(queue);
  clSetKernelArg(memset_kernel, 0, sizeof(in2), (void*) &in2);
  memset_global_work_size = max_in2_index - min_in2_index;
  retval = clEnqueueNDRangeKernel(queue, memset_kernel, 1, NULL, &memset_global_work_size, NULL, 0, NULL, NULL);
  if(retval != CL_SUCCESS) {
      switch(retval) {
      CASE(CL_INVALID_PROGRAM_EXECUTABLE);
      CASE(CL_INVALID_COMMAND_QUEUE);
      CASE(CL_INVALID_KERNEL);
      }
      assert(false &&  "could not enqueue memset kernel the second time");
  }
  clFlush(queue);

  ptr = (cl_float *) clEnqueueMapBuffer( queue,
                                        in1,
                                        CL_TRUE,
                                        CL_MAP_READ,
                                        0,
                                        (max_in1_index - min_in1_index) * sizeof(cl_float),
                                        0, NULL, NULL, NULL );

  for(size_t i=0; i < (max_in1_index - min_in1_index); i++) {
      if(in1_before[i] == ptr[i] && i != ptr[i]) {
          std::cerr << "Error: ptr[" << i << "] != " << i << ", instead it was: " << ptr[i] << ", and before it was: " << in1_before[i] << std::endl;
          assert(false);
      }
  }

  ptr = (cl_float *) clEnqueueMapBuffer( queue,
                                        in2,
                                        CL_TRUE,
                                        CL_MAP_READ,
                                        0,
                                        (max_in2_index - min_in2_index) * sizeof(cl_float),
                                        0, NULL, NULL, NULL );

  for(size_t i=0; i < (max_in2_index - min_in2_index); i++) {
      if(i != ptr[i]) {
          std::cerr << "Error: ptr[" << i << "] != " << i << std::endl;
          assert(false);
      }
  }
  // 6. Launch the kernel. Let OpenCL pick the local work size.
  size_t global_work_size[] = {1792, 8};
  size_t local_work_size[] = {256, 1};
  clSetKernelArg(kernel, 0, sizeof(x_t199), (void*) &x_t199);
  clSetKernelArg(kernel, 1, sizeof(in1), (void*) &in1);
  clSetKernelArg(kernel, 2, sizeof(in2), (void*) &in2);

  std::cout << "Running kernel" << std::endl;
  clEnqueueNDRangeKernel( queue,
                          kernel,
                          2,
                          NULL,
                          global_work_size,
                          local_work_size,
                          0,
                          NULL, NULL);

  std::cout << "waiting for kernel" << std::endl;
  clFinish( queue );

  std::cout << "mapping results" << std::endl;
  // 7. Look at the results via synchronous buffer map.
  ptr = (cl_float *) clEnqueueMapBuffer( queue,
                                        x_t199,
                                        CL_TRUE,
                                        CL_MAP_READ,
                                        0,
                                        (max_x_t199_index - min_x_t199_index) * sizeof(cl_float),
                                        0, NULL, NULL, NULL );

  for(size_t i=0; i < (max_x_t199_index - min_x_t199_index); i++)
      printf("%lu %f\n", i, ptr[i]);

  return 0;
}
