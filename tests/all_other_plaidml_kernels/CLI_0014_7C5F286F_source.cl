#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 80 20000 128
// lid: 1 2 128
__kernel void kernel_c6_sdk_10(__global float* restrict  out, __global const float* restrict  data, __global const int* restrict  idx)
{
  int gidx0 = get_global_id(0);
  int gidx1 = get_global_id(1);
  int gidx2 = get_global_id(2);
  out[((gidx0 * 128) + gidx2)] = data[((clamp((int)idx[gidx0], (int)0, (int)19999) * 128) + (int)gidx2)];
}
