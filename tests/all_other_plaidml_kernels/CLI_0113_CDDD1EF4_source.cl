#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Original:
// X_T3486[x0, y1 : _T4221, _T4222] = +(X_T3485[x0, z] * X_I_9[z, y1])
// With Index Variables Made Integral:
// X_T3486[x0, y1 : _T4221, _T4222] = +(X_T3485[x0, z] * X_I_9[z, y1]), 500000000 + x0 < 1000000000, 500000000 + y1 < 1000000000, 500000000 + z < 1000000000
// Constraints:{ 0 <= x0 < 1, 0 <= y1 < 1, 0 <= x0 < 1, 0 <= y1 < 1, 0 <= z < 128, 0 <= z < 128, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + y1 < 1000000000, 0 <= 500000000 + z < 1000000000 }
// Merged Parallel Constraints:{ 0 <= x0 < 1, 0 <= y1 < 1, 0 <= z < 128 }
// Defracted:
// X_T3486[x0, y1 : _T4221, _T4222] = +(X_T3485[x0, z] * X_I_9[z, y1]), 500000000 + x0 < 1000000000, 500000000 + y1 < 1000000000, 500000000 + z < 1000000000
// Flattened:
//              Range   X_T3486   X_T3485     X_I_9  
//        z       128         0         1         1  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { z }
// Ranges: { 128 }
// Out stride: { 0 }
// Input 1 offset: 0
// Input 1 stride: { 1 }
// Input 2 offset: 0
// Input 2 stride: { 1 }
// Elementwise input X_I_8 shape: fp32(1):(1):4 bytes
// Elementwise op: [[pid(Add)]] X_T3487 = add(X_T3486, X_I_8)
// Elementwise op: X_T3493 = neg(X_T3487)
// Elementwise op: X_T3494 = exp(X_T3493)
// Elementwise op: X_T3495 = add(X_T3492, X_T3494)
// Elementwise op: X_T3488 = div(X_T3491, X_T3495)
// Tile size: { 128 }
// Contraction output var shape: fp32(1, 1):(1, 1):4 bytes
// Computed true ops: 896
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 2048
// Computed out regs: 1024
// Computed mem read: 1028
// Computed mem write: 128
// Computed operations: 128
// Computed rollups: 7
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c6_sdk_748(__global float* restrict  X_T3488, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_8)
{
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[128];
  __local float in2_shared[128];
  for (int z_gid = 0; z_gid < 128; z_gid += 128)
  {
    {
      int z_tid = (tid % 128);
      int gidx = (z_gid + z_tid);
      in1_shared[z_tid] = in1[clamp((int)gidx, (int)0, (int)127)];
    }
    {
      int z_tid = (tid % 128);
      int gidx = (z_gid + z_tid);
      in2_shared[z_tid] = in2[clamp((int)gidx, (int)0, (int)127)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int z_tid = (tid % 128);
    float val1 = in1_shared[z_tid];
    float val2 = in2_shared[z_tid];
    float agg_rhs = mad(val2, val1, agg[0]);
    agg[0] = agg_rhs;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  __local float merge_shared[128];
  {
    merge_shared[tid] = agg[0];
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
    if ((tid < 16))
    {
      merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 16)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 8))
    {
      merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 8)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 4))
    {
      merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 4)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 2))
    {
      merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 2)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 1))
    {
      merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 1)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 1))
    {
      agg[0] = merge_shared[tid];
    }
  }
  if ((tid < 1))
  {
    float LX_T3486 = agg[0];
    float LX_I_8 = X_I_8[0];
    float LX_T3487 = (LX_T3486 + LX_I_8);
    float LX_T3493 = (-LX_T3487);
    float LX_T3494 = native_exp(LX_T3493);
    float LX_T3495 = (1.0f + LX_T3494);
    float LX_T3488 = (1.0f / LX_T3495);
    X_T3488[0] = LX_T3488;
  }
}
