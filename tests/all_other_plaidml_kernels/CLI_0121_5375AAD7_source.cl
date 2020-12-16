#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2048 1 1
// lid: 256 1 1
__kernel void kernel_c81_sdk_0(__global float* restrict  out, __global uint* restrict  state_out, __global const uint* restrict  state_in)
{
  int i = get_global_id(0);
  uint s1 = state_in[i];
  uint s2 = state_in[(i + 2048)];
  uint s3 = state_in[(i + 4096)];
  while ((i < 819200))
  {
    s1 = (uint)((((long)s1 & 4294967294) << 12) ^ (long)(((s1 << (uint)13) ^ s1) >> (uint)19));
    s2 = (uint)((((long)s2 & 4294967288) << 4) ^ (long)(((s2 << (uint)2) ^ s2) >> (uint)25));
    s3 = (uint)((((long)s3 & 4294967280) << 17) ^ (long)(((s3 << (uint)3) ^ s3) >> (uint)11));
    out[i] = ((float)((s1 ^ s2) ^ s3) / 4294967296.0f);
    i = (i + 2048);
  }
  i = get_global_id(0);
  state_out[i] = s1;
  state_out[(i + 2048)] = s2;
  state_out[(i + 4096)] = s3;
}
