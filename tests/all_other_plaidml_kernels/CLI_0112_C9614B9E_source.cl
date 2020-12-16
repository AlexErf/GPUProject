#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 1 1
// lid: 256 1 1
// Original:
// X_T3479[x0, y1 : _T4213, _T4214] = +(X_T3447[x0, z] * X_T397[z, y1])
// With Index Variables Made Integral:
// X_T3479[x0, y1 : _T4213, _T4214] = +(X_T3447[x0, z] * X_T397[z, y1]), 500000000 + x0 < 1000000000, 500000000 + y1 < 1000000000, 500000000 + z < 1000000000
// Constraints:{ 0 <= x0 < 1, 0 <= x0 < 1, 0 <= y1 < 128, 0 <= z < 128, 0 <= z < 128, 0 <= y1 < 128, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + y1 < 1000000000, 0 <= 500000000 + z < 1000000000 }
// Merged Parallel Constraints:{ 0 <= x0 < 1, 0 <= y1 < 128, 0 <= z < 128 }
// Defracted:
// X_T3479[x0, y1 : _T4213, _T4214] = +(X_T3447[x0, z] * X_T397[z, y1]), 500000000 + x0 < 1000000000, 500000000 + y1 < 1000000000, 500000000 + z < 1000000000
// Flattened:
//              Range   X_T3479   X_T3447    X_T397  
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
// Elementwise input X_T3478 shape: fp32(1, 128):(128, 1):512 bytes
// Elementwise input X_T3476 shape: fp32(1, 128):(128, 1):512 bytes
// Elementwise input X_T3466 shape: fp32(1, 128):(128, 1):512 bytes
// Elementwise input X_T3455 shape: fp32(1, 128):(128, 1):512 bytes
// Elementwise op: [[pid(Add)]] X_T3480 = add(X_T3478, X_T3479)
// Elementwise op: [[pid(Tanh)]] X_T3481 = tanh(X_T3480)
// Elementwise op: [[pid(Mul)]] X_T3482 = mul(X_T3476, X_T3481)
// Elementwise op: [[pid(Add)]] X_T3483 = add(X_T3466, X_T3482)
// Elementwise op: [[pid(Tanh)]] X_T3484 = tanh(X_T3483)
// Elementwise op: [[pid(Mul)]] X_T3485 = mul(X_T3455, X_T3484)
// Tile size: { 32, 128 }
// Contraction output var shape: fp32(1, 128):(128, 1):512 bytes
// Computed true ops: 131072
// Computed work groups: 4
// Computed inner loops: 1
// Computed shared mem: 18048
// Computed out regs: 1024
// Computed mem read: 16912
// Computed mem write: 128
// Computed operations: 256
// Computed rollups: 3
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1024, 1, 1
__kernel void kernel_c6_sdk_747(__global float* restrict  X_T3485, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_T3478, __global const float* restrict  X_T3476, __global const float* restrict  X_T3466, __global const float* restrict  X_T3455)
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
    float LX_T3479 = agg[0];
    int gout_idx = (y1_gid + y1_tid);
    float LX_T3478 = X_T3478[gout_idx];
    float LX_T3476 = X_T3476[gout_idx];
    float LX_T3466 = X_T3466[gout_idx];
    float LX_T3455 = X_T3455[gout_idx];
    float LX_T3480 = (LX_T3478 + LX_T3479);
    float LX_T3481 = tanh(LX_T3480);
    float LX_T3482 = (LX_T3476 * LX_T3481);
    float LX_T3483 = (LX_T3466 + LX_T3482);
    float LX_T3484 = tanh(LX_T3483);
    float LX_T3485 = (LX_T3455 * LX_T3484);
    X_T3485[gout_idx] = LX_T3485;
  }
}
