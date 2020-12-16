#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 42 1
// lid: 256 1 1
// Original:
// X_T483[n, x0, x1, c : _T723, _T724, _T725, _T726] = +(X_T482[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1
// With Index Variables Made Integral:
// X_T483[n, x0, x1, c : _T723, _T724, _T725, _T726] = +(X_T482[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= k0 < 1, 0 <= k1 < 1, 0 <= n < 1, 0 <= n < 1, 0 <= x0 < 42, 0 <= x1 < 42, 0 <= k0 + 2*x0 < 83, 0 <= k1 + 2*x1 < 83, 0 <= c < 168, 0 <= c < 168, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= k0 < 1, 0 <= k1 < 1, 0 <= n < 1, 0 <= x0 < 42, 0 <= x1 < 42, 0 <= k0 + 2*x0 < 83, 0 <= k1 + 2*x1 < 83, 0 <= c < 168 }
// Defracted:
// X_T483[n, x0, x1, c : _T723, _T724, _T725, _T726] = +(X_T482[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T483    X_T482  
//        c       168         1         1  
//       x0        42      7056     27888  
//       x1        42       168       336  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, x0, x1 }
// Ranges: { 168, 42, 42 }
// Out stride: { 1, 7056, 168 }
// Input 1 offset: 0
// Input 1 stride: { 1, 27888, 336 }
// Elementwise input X_T481 shape: fp32(1, 42, 42, 168):(296352, 7056, 168, 1):1157.62 KiB
// Elementwise op: [[pid(normal_A_block_0)]] X_T484 = div(X_T481, X_T483)
// Tile size: { 168, 2, 1 }
// Contraction output var shape: fp32(1, 42, 42, 168):(296352, 7056, 168, 1):1157.62 KiB
// Computed true ops: 889056
// Computed work groups: 882
// Computed inner loops: 1
// Computed shared mem: 1352
// Computed out regs: 2048
// Computed mem read: 1328
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 42, 1
__kernel void kernel_c42_sdk_168(__global float* restrict  X_T484, __global const float* restrict  in1, __global const float* restrict  X_T481)
{
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[338];
  int x1_gid = get_group_id(1);
  int x0_gid = (get_group_id(0) * 2);
  {
    {
      int gbase = ((x1_gid * 336) + (x0_gid * 27888));
      int c_tid = (tid % 256);
      int c_cond = (c_tid < 168);
      if (c_cond)
      {
        for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
        {
          int lidx = (c_tid + (169 * x0_lid));
          int gidx = ((gbase + c_tid) + (27888 * x0_lid));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)1157351)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int c_tid = (tid % 128);
    int x0_tid = ((tid / 128) % 2);
    for (int c_lid = 0; c_lid < 2; c_lid += 1)
    {
      int c_cond = ((c_lid < 1) || (c_tid < 40));
      int c = select((int)0, (int)((128 * c_lid) + c_tid), (int)c_cond);
      float val1 = in1_shared[(c + (169 * x0_tid))];
      float agg_rhs = (agg[c_lid] + val1);
      agg[c_lid] = select((float)agg[c_lid], (float)agg_rhs, (int)c_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 128);
  int x0_tid = ((tid / 128) % 2);
  for (int c_lid = 0; c_lid < 2; c_lid += 1)
  {
    int c_cond = ((c_lid < 1) || (c_tid < 40));
    if (c_cond)
    {
      int c = ((128 * c_lid) + c_tid);
      float LX_T483 = agg[c_lid];
      int gout_idx = ((c + (7056 * (x0_gid + x0_tid))) + (168 * x1_gid));
      float LX_T481 = X_T481[gout_idx];
      float LX_T484 = (LX_T481 / LX_T483);
      X_T484[gout_idx] = LX_T484;
    }
  }
}
