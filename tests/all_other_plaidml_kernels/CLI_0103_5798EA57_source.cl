#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 1 1
// lid: 256 1 1
// Original:
// X_T355[x0, y1 : _T524, _T525] = +(X_T354[x0, z] * X_T32[z, y1])
// With Index Variables Made Integral:
// X_T355[x0, y1 : _T524, _T525] = +(X_T354[x0, z] * X_T32[z, y1]), 500000000 + x0 < 1000000000, 500000000 + y1 < 1000000000, 500000000 + z < 1000000000
// Constraints:{ 0 <= x0 < 1, 0 <= x0 < 1, 0 <= y1 < 128, 0 <= z < 128, 0 <= z < 128, 0 <= y1 < 128, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + y1 < 1000000000, 0 <= 500000000 + z < 1000000000 }
// Merged Parallel Constraints:{ 0 <= x0 < 1, 0 <= y1 < 128, 0 <= z < 128 }
// Defracted:
// X_T355[x0, y1 : _T524, _T525] = +(X_T354[x0, z] * X_T32[z, y1]), 500000000 + x0 < 1000000000, 500000000 + y1 < 1000000000, 500000000 + z < 1000000000
// Flattened:
//              Range    X_T355    X_T354     X_T32  
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
// Elementwise input X_T348 shape: fp32(1, 128):(128, 1):512 bytes
// Elementwise op: [[pid(Add)]] X_T356 = add(X_T348, X_T355)
// Elementwise op: X_T357 = cmp_lt(X_T356, X_T21)
// Elementwise op: X_T359 = cmp_gt(X_T356, X_T358)
// Elementwise op: X_T362 = mul(X_T361, X_T356)
// Elementwise op: X_T363 = add(X_T360, X_T362)
// Elementwise op: X_T364 = cond(X_T359, X_I_17_0, X_T363)
// Elementwise op: [[pid(HardSigmoid)]] X_T365 = cond(X_T357, X_T4, X_T364)
// Tile size: { 32, 128 }
// Contraction output var shape: fp32(1, 128):(128, 1):512 bytes
// Computed true ops: 147456
// Computed work groups: 4
// Computed inner loops: 1
// Computed shared mem: 18048
// Computed out regs: 1024
// Computed mem read: 16900
// Computed mem write: 128
// Computed operations: 256
// Computed rollups: 3
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1024, 1, 1
__kernel void kernel_c6_sdk_179(__global float* restrict  X_T365, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_T348)
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
    float LX_T355 = agg[0];
    int gout_idx = (y1_gid + y1_tid);
    float LX_T348 = X_T348[gout_idx];
    float LX_T356 = (LX_T348 + LX_T355);
    int LX_T357 = (LX_T356 < -2.5f);
    int LX_T359 = (LX_T356 > 2.5f);
    float LX_T362 = (0.20000000298023224f * LX_T356);
    float LX_T363 = (0.5f + LX_T362);
    float LX_T364 = select((float)LX_T363, (float)1, (int)LX_T359);
    float LX_T365 = select((float)LX_T364, (float)0, (int)LX_T357);
    X_T365[gout_idx] = LX_T365;
  }
}
