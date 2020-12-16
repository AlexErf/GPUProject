#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4096 1 1
// lid: 256 1 1
// Original:
// X_T450[n, x0, x1, co : _T515, _T516, _T517, _T518] = +(X_T449[n, k0 + x0, k1 + x1, ci] * X_I_2[k0, k1, ci, co])
// With Index Variables Made Integral:
// X_T450[n, x0, x1, co : _T515, _T516, _T517, _T518] = +(X_T449[n, k0 + x0, k1 + x1, ci] * X_I_2[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= x0 < 1, 0 <= x1 < 1, 0 <= n < 1, 0 <= k0 + x0 < 1, 0 <= k1 + x1 < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 1000, 0 <= co < 1000, 0 <= ci < 1024, 0 <= ci < 1024, 0 <= 500000000 + ci < 1000000000, 0 <= 500000000 + co < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= x0 < 1, 0 <= x1 < 1, 0 <= k0 + x0 < 1, 0 <= k1 + x1 < 1, 0 <= k0 < 1, 0 <= k1 < 1, 0 <= co < 1000, 0 <= ci < 1024 }
// Defracted:
// X_T450[n, x0, x1, co : _T515, _T516, _T517, _T518] = +(X_T449[n, k0 + x0, k1 + x1, ci] * X_I_2[k0, k1, ci, co]), 500000000 + ci < 1000000000, 500000000 + co < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T450    X_T449     X_I_2  
//       ci      1024         0         1      1000  
//       co      1000         1         0         1  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { ci, co }
// Ranges: { 1024, 1000 }
// Out stride: { 0, 1 }
// Input 1 offset: 0
// Input 1 stride: { 1, 0 }
// Input 2 offset: 0
// Input 2 stride: { 1000, 1 }
// Elementwise input X_I_1 shape: fp32(1000):(1):3.90625 KiB
// Elementwise op: [[pid(Add)]] X_T451 = add(X_T450, X_I_1)
// Tile size: { 64, 64 }
// Contraction output var shape: fp32(1, 1, 1, 1000):(1000, 1000, 1000, 1):3.90625 KiB
// Computed true ops: 3072000
// Computed work groups: 16
// Computed inner loops: 16
// Computed shared mem: 17920
// Computed out regs: 1024
// Computed mem read: 16648
// Computed mem write: 256
// Computed operations: 256
// Computed rollups: 2
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4096, 1, 1
__kernel void kernel_c25_sdk_114(__global float* restrict  X_T451, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_1)
{
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[64];
  __local float in2_shared[4160];
  int co_gid = (get_group_id(0) * 64);
  for (int ci_gid = 0; ci_gid < 1024; ci_gid += 64)
  {
    {
      int ci_tid = (tid % 64);
      if ((tid < 64))
      {
        int gidx = (ci_gid + ci_tid);
        in1_shared[ci_tid] = in1[clamp((int)gidx, (int)0, (int)1023)];
      }
    }
    {
      int gbase = (co_gid + (ci_gid * 1000));
      int co_tid = (tid % 64);
      int ci_tid = ((tid / 64) % 4);
      for (int ci_lid = 0; ci_lid < 16; ci_lid += 1)
      {
        int ci = ((4 * ci_lid) + ci_tid);
        int lidx = (co_tid + (65 * ci));
        int gidx = ((gbase + co_tid) + (1000 * ci));
        in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)1023999)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int ci_tid = ((tid / 64) % 4);
    for (int ci_lid = 0; ci_lid < 16; ci_lid += 1)
    {
      int ci = ((4 * ci_lid) + ci_tid);
      int co_tid = (tid % 64);
      float val1 = in1_shared[ci];
      float val2 = in2_shared[(co_tid + (65 * ci))];
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
    if ((tid < 64))
    {
      agg[0] = merge_shared[tid];
    }
  }
  int co_tid = (tid % 64);
  int co_cond = ((co_gid != 960) || (co_tid < 40));
  if (co_cond)
  {
    if ((tid < 64))
    {
      float LX_T450 = agg[0];
      int gout_idx = (co_gid + co_tid);
      float LX_I_1 = X_I_1[gout_idx];
      float LX_T451 = (LX_T450 + LX_I_1);
      X_T451[gout_idx] = LX_T451;
    }
  }
}
