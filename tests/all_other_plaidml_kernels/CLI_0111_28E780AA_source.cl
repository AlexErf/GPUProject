#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 1 1
// lid: 256 1 1
// Original:
// X_T398[x0, y1 : _T579, _T580] = +(X_T354[x0, z] * X_T397[z, y1])
// With Index Variables Made Integral:
// X_T398[x0, y1 : _T579, _T580] = +(X_T354[x0, z] * X_T397[z, y1]), 500000000 + x0 < 1000000000, 500000000 + y1 < 1000000000, 500000000 + z < 1000000000
// Constraints:{ 0 <= x0 < 1, 0 <= x0 < 1, 0 <= y1 < 128, 0 <= z < 128, 0 <= z < 128, 0 <= y1 < 128, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + y1 < 1000000000, 0 <= 500000000 + z < 1000000000 }
// Merged Parallel Constraints:{ 0 <= x0 < 1, 0 <= y1 < 128, 0 <= z < 128 }
// Defracted:
// X_T398[x0, y1 : _T579, _T580] = +(X_T354[x0, z] * X_T397[z, y1]), 500000000 + x0 < 1000000000, 500000000 + y1 < 1000000000, 500000000 + z < 1000000000
// Flattened:
//              Range    X_T398    X_T354    X_T397  
//       y1       128         1         0         1  
//        z       128         0         1       128  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { y1, z }
// Ranges: { 128, 128 }
// Out stride: { 1, 0 }
// Input 1 offset: 0
// Input 1 stride: { 0, 1 }
// Input 2 offset: 0
// Input 2 stride: { 1, 128 }
// Elementwise input X_T396 shape: fp32(1, 128):(128, 1):512 bytes
// Elementwise input X_T392 shape: fp32(1, 128):(128, 1):512 bytes
// Elementwise input X_T379 shape: fp32(1, 128):(128, 1):512 bytes
// Elementwise input X_T365 shape: fp32(1, 128):(128, 1):512 bytes
// Elementwise input X_T353 shape: fp32(1, 128):(128, 1):512 bytes
// Elementwise op: [[pid(Add)]] X_T399 = add(X_T396, X_T398)
// Elementwise op: [[pid(Tanh)]] X_T400 = tanh(X_T399)
// Elementwise op: [[pid(Mul)]] X_T401 = mul(X_T392, X_T400)
// Elementwise op: [[pid(Add)]] X_T402 = add(X_T379, X_T401)
// Elementwise op: [[pid(Tanh)]] X_T403 = tanh(X_T402)
// Elementwise op: [[pid(Mul)]] X_T404 = mul(X_T365, X_T403)
// Elementwise op: [[pid(Mul)]] X_T405 = mul(X_T404, X_T353)
// Tile size: { 32, 128 }
// Contraction output var shape: fp32(1, 128):(128, 1):512 bytes
// Computed true ops: 147456
// Computed work groups: 4
// Computed inner loops: 1
// Computed shared mem: 18048
// Computed out regs: 1024
// Computed mem read: 16916
// Computed mem write: 256
// Computed operations: 256
// Computed rollups: 3
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1024, 1, 1
__kernel void kernel_c6_sdk_194(__global float* restrict  X_T402, __global float* restrict  X_T405, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_T396, __global const float* restrict  X_T392, __global const float* restrict  X_T379, __global const float* restrict  X_T365, __global const float* restrict  X_T353)
{
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[128];
  __local float in2_shared[4128];
  int y1_gid = (get_group_id(0) * 32);
  for (int z_gid = 0; z_gid < 128; z_gid += 128)
  {
    {
      int z_tid = (tid % 128);
      if ((tid < 128))
      {
        int gidx = (z_gid + z_tid);
        in1_shared[z_tid] = in1[clamp((int)gidx, (int)0, (int)127)];
      }
    }
    {
      int gbase = (y1_gid + (z_gid * 128));
      int y1_tid = (tid % 32);
      int z_tid = ((tid / 32) % 8);
      for (int z_lid = 0; z_lid < 16; z_lid += 1)
      {
        int z = ((8 * z_lid) + z_tid);
        int lidx = ((129 * y1_tid) + z);
        int gidx = ((gbase + y1_tid) + (128 * z));
        in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)16383)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int z_tid = ((tid / 32) % 8);
    for (int z_lid = 0; z_lid < 16; z_lid += 1)
    {
      int z = ((8 * z_lid) + z_tid);
      int y1_tid = (tid % 32);
      float val1 = in1_shared[z];
      float val2 = in2_shared[((129 * y1_tid) + z)];
      float agg_rhs = mad(val2, val1, agg[0]);
      agg[0] = agg_rhs;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  __local float merge_shared[256];
  {
    merge_shared[tid] = agg[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 128))
    {
      merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 128)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 64))
    {
      merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 64)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 32))
    {
      merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 32)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 32))
    {
      agg[0] = merge_shared[tid];
    }
  }
  int y1_tid = (tid % 32);
  if ((tid < 32))
  {
    float LX_T398 = agg[0];
    int gout_idx = (y1_gid + y1_tid);
    float LX_T396 = X_T396[gout_idx];
    float LX_T392 = X_T392[gout_idx];
    float LX_T379 = X_T379[gout_idx];
    float LX_T365 = X_T365[gout_idx];
    float LX_T353 = X_T353[gout_idx];
    float LX_T399 = (LX_T396 + LX_T398);
    float LX_T400 = tanh(LX_T399);
    float LX_T401 = (LX_T392 * LX_T400);
    float LX_T402 = (LX_T379 + LX_T401);
    float LX_T403 = tanh(LX_T402);
    float LX_T404 = (LX_T365 * LX_T403);
    float LX_T405 = (LX_T404 * LX_T353);
    X_T402[gout_idx] = LX_T402;
    X_T405[gout_idx] = LX_T405;
  }
}
