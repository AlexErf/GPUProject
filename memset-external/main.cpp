//
// Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//

// A minimalist OpenCL program.

#include <CL/cl.h>
#include <stdio.h>

#include <unistd.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

#define NWITEMS 512
// A simple memset kernel
const char *source =
"kernel void memset(   global uint *dst )             \n"
"{                                                    \n"
"    dst[get_global_id(0)] = get_global_id(0);        \n"
"}                                                    \n";
off_t fsize(const char *filename) {
    struct stat st;

    if (stat(filename, &st) == 0)
        return st.st_size;

    return -1;
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
  while(1) {
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


  /* cl_program program = clCreateProgramWithSource( context, */
  /*                                                 1, */
  /*                                                 &source, */
  /*                                                 NULL, NULL ); */

  std::cout << "Building program" << std::endl;
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
      //exit(1);
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

  std::cout << "Building kernel" << std::endl;
  cl_kernel kernel = clCreateKernel( program, "memset", NULL );

  // 5. Create a data buffer.
  cl_mem buffer = clCreateBuffer( context,
                                  CL_MEM_WRITE_ONLY,
                                  NWITEMS * sizeof(cl_uint),
                                  NULL, NULL );

  // 6. Launch the kernel. Let OpenCL pick the local work size.
  size_t global_work_size = NWITEMS;
  clSetKernelArg(kernel, 0, sizeof(buffer), (void*) &buffer);

  std::cout << "Running kernel" << std::endl;
  clEnqueueNDRangeKernel( queue,
                          kernel,
                          1,
                          NULL,
                          &global_work_size,
                          NULL,
                          0,
                          NULL, NULL);

  std::cout << "waiting for kernel" << std::endl;
  clFinish( queue );

  std::cout << "mapping results" << std::endl;
  // 7. Look at the results via synchronous buffer map.
  cl_uint *ptr;
  ptr = (cl_uint *) clEnqueueMapBuffer( queue,
                                        buffer,
                                        CL_TRUE,
                                        CL_MAP_READ,
                                        0,
                                        NWITEMS * sizeof(cl_uint),
                                        0, NULL, NULL, NULL );

  int i;

  for(i=0; i < NWITEMS; i++)
      printf("%d %d\n", i, ptr[i]);

  }
  return 0;
}
