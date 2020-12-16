#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 9 1
// lid: 256 1 1
// Original:
// X_T1416[n, x0, x1, c : _T2224, _T2225, _T2226, _T2227] = +(X_T1415[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1
// With Index Variables Made Integral:
// X_T1416[n, x0, x1, c : _T2224, _T2225, _T2226, _T2227] = +(X_T1415[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= k0 < 1, 0 <= k1 < 1, 0 <= n < 1, 0 <= n < 1, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + 2*x0 < 28, 0 <= k1 + 2*x1 < 28, 0 <= c < 264, 0 <= c < 264, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= k0 < 1, 0 <= k1 < 1, 0 <= n < 1, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + 2*x0 < 28, 0 <= k1 + 2*x1 < 28, 0 <= c < 264 }
// Defracted:
// X_T1416[n, x0, x1, c : _T2224, _T2225, _T2226, _T2227] = +(X_T1415[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T1416   X_T1415  
//        c       264         1         1  
//       x0        14      3696     14784  
//       x1        14       264       528  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, x0, x1 }
// Ranges: { 264, 14, 14 }
// Out stride: { 1, 3696, 264 }
// Input 1 offset: 0
// Input 1 stride: { 1, 14784, 528 }
// Elementwise input X_T1414 shape: fp32(1, 14, 14, 264):(51744, 3696, 264, 1):202.125 KiB
// Elementwise op: [[pid(normal_A_block_5)]] X_T1417 = div(X_T1414, X_T1416)
// Tile size: { 32, 2, 14 }
// Contraction output var shape: fp32(1, 14, 14, 264):(51744, 3696, 264, 1):202.125 KiB
// Computed true ops: 155232
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 3704
// Computed out regs: 4096
// Computed mem read: 3696
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 9, 1
__kernel void kernel_c42_sdk_534(__global float* restrict  X_T1417, __global const float* restrict  in1, __global const float* restrict  X_T1414)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[926];
  int c_gid = (get_group_id(1) * 32);
  int x0_gid = (get_group_id(0) * 2);
  {
    {
      int gbase = (c_gid + (x0_gid * 14784));
      int c_tid = (tid % 32);
      int x0_tid = ((tid / 32) % 2);
      int x1_tid = ((tid / 64) % 4);
      for (int x1_lid = 0; x1_lid < 4; x1_lid += 1)
      {
        int x1_cond = ((x1_lid < 3) || (x1_tid < 2));
        if (x1_cond)
        {
          int x1 = ((4 * x1_lid) + x1_tid);
          int lidx = ((c_tid + (463 * x0_tid)) + (33 * x1));
          int gidx = (((gbase + c_tid) + (14784 * x0_tid)) + (528 * x1));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)206975)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int c_tid = (tid % 32);
    int x1_tid = ((tid / 32) % 4);
    int x0_tid = ((tid / 128) % 2);
    for (int x1_lid = 0; x1_lid < 4; x1_lid += 1)
    {
      int x1_cond = ((x1_lid < 3) || (x1_tid < 2));
      int x1 = select((int)0, (int)((4 * x1_lid) + x1_tid), (int)x1_cond);
      float val1 = in1_shared[((c_tid + (33 * x1)) + (463 * x0_tid))];
      float agg_rhs = (agg[x1_lid] + val1);
      agg[x1_lid] = select((float)agg[x1_lid], (float)agg_rhs, (int)x1_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  int c_cond = ((c_gid != 256) || (c_tid < 8));
  if (c_cond)
  {
    for (int x1_lid = 0; x1_lid < 4; x1_lid += 1)
    {
      int x1_cond = ((x1_lid < 3) || (x1_tid < 2));
      if (x1_cond)
      {
        int x1 = ((4 * x1_lid) + x1_tid);
        float LX_T1416 = agg[x1_lid];
        int gout_idx = (((c_gid + c_tid) + (3696 * (x0_gid + x0_tid))) + (264 * x1));
        float LX_T1414 = X_T1414[gout_idx];
        float LX_T1417 = (LX_T1414 / LX_T1416);
        X_T1417[gout_idx] = LX_T1417;
      }
    }
  }
}
