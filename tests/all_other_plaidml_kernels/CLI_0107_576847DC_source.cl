#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 83 1
// lid: 256 1 1
// Original:
// X_T261[n, x0, x1, c : _T360, _T361, _T362, _T363] = +(X_T260[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1
// With Index Variables Made Integral:
// X_T261[n, x0, x1, c : _T360, _T361, _T362, _T363] = +(X_T260[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= k0 < 1, 0 <= k1 < 1, 0 <= n < 1, 0 <= n < 1, 0 <= x0 < 83, 0 <= x1 < 83, 0 <= c < 96, 0 <= c < 96, 0 <= k0 + 2*x0 < 165, 0 <= k1 + 2*x1 < 165, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= k0 < 1, 0 <= k1 < 1, 0 <= n < 1, 0 <= x0 < 83, 0 <= x1 < 83, 0 <= c < 96, 0 <= k0 + 2*x0 < 165, 0 <= k1 + 2*x1 < 165 }
// Defracted:
// X_T261[n, x0, x1, c : _T360, _T361, _T362, _T363] = +(X_T260[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T261    X_T260  
//        c        96         1         1  
//       x0        83      7968     31680  
//       x1        83        96       192  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, x0, x1 }
// Ranges: { 96, 83, 83 }
// Out stride: { 1, 7968, 96 }
// Input 1 offset: 0
// Input 1 stride: { 1, 31680, 192 }
// Elementwise input X_T259 shape: fp32(1, 83, 83, 96):(661344, 7968, 96, 1):2583.38 KiB
// Elementwise op: [[pid(reduction_A_block_stem_2)]] X_T262 = div(X_T259, X_T261)
// Tile size: { 96, 4, 1 }
// Contraction output var shape: fp32(1, 83, 83, 96):(661344, 7968, 96, 1):2583.38 KiB
// Computed true ops: 1984032
// Computed work groups: 1743
// Computed inner loops: 1
// Computed shared mem: 1552
// Computed out regs: 2048
// Computed mem read: 1584
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 83, 1
__kernel void kernel_c42_sdk_83(__global float* restrict  X_T262, __global const float* restrict  in1, __global const float* restrict  X_T259)
{
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[388];
  int x1_gid = get_group_id(1);
  int x0_gid = (get_group_id(0) * 4);
  {
    {
      int gbase = ((x1_gid * 192) + (x0_gid * 31680));
      int c_tid = (tid % 128);
      int x0_tid = ((tid / 128) % 2);
      int c_cond = (c_tid < 96);
      if (c_cond)
      {
        for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          int lidx = (c_tid + (97 * x0));
          int gidx = ((gbase + c_tid) + (31680 * x0));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)2613599)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int c_tid = (tid % 64);
    int x0_tid = ((tid / 64) % 4);
    for (int c_lid = 0; c_lid < 2; c_lid += 1)
    {
      int c_cond = ((c_lid < 1) || (c_tid < 32));
      int c = select((int)0, (int)((64 * c_lid) + c_tid), (int)c_cond);
      float val1 = in1_shared[(c + (97 * x0_tid))];
      float agg_rhs = (agg[c_lid] + val1);
      agg[c_lid] = select((float)agg[c_lid], (float)agg_rhs, (int)c_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 64);
  int x0_tid = ((tid / 64) % 4);
  int x0_cond = ((x0_gid != 80) || (x0_tid < 3));
  if (x0_cond)
  {
    for (int c_lid = 0; c_lid < 2; c_lid += 1)
    {
      int c_cond = ((c_lid < 1) || (c_tid < 32));
      if (c_cond)
      {
        int c = ((64 * c_lid) + c_tid);
        float LX_T261 = agg[c_lid];
        int gout_idx = ((c + (7968 * (x0_gid + x0_tid))) + (96 * x1_gid));
        float LX_T259 = X_T259[gout_idx];
        float LX_T262 = (LX_T259 / LX_T261);
        X_T262[gout_idx] = LX_T262;
      }
    }
  }
}
