#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2816 11 1
// lid: 256 1 1
// Original:
// X_T320[n, x0, x1, c : _T477, _T478, _T479, _T480] = +(X_T319[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T320[n, x0, x1, c : _T477, _T478, _T479, _T480] = +(X_T319[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 42, 0 <= x1 < 42, 0 <= c < 84, 0 <= c < 84, 0 <= k0 + 2*x0 < 85, 0 <= k1 + 2*x1 < 85, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 42, 0 <= x1 < 42, 0 <= c < 84, 0 <= k0 + 2*x0 < 85, 0 <= k1 + 2*x1 < 85 }
// Defracted:
// X_T320[n, x0, x1, c : _T477, _T478, _T479, _T480] = +(X_T319[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T320    X_T319  
//        c        84         1         1  
//       k0         3         0      7140  
//       k1         3         0        84  
//       x0        42      3528     14280  
//       x1        42        84       168  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 84, 3, 3, 42, 42 }
// Out stride: { 1, 0, 0, 3528, 84 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7140, 84, 14280, 168 }
// Elementwise input X_T318 shape: fp32(1, 42, 42, 84):(148176, 3528, 84, 1):578.812 KiB
// Elementwise op: [[pid(reduction_A_block_stem_2)]] X_T321 = div(X_T318, X_T320)
// Tile size: { 84, 3, 3, 4, 4 }
// Contraction output var shape: fp32(1, 42, 42, 84):(148176, 3528, 84, 1):578.812 KiB
// Computed true ops: 4000752
// Computed work groups: 121
// Computed inner loops: 1
// Computed shared mem: 27216
// Computed out regs: 6144
// Computed mem read: 27328
// Computed mem write: 6144
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2816, 11, 1
__kernel void kernel_c42_sdk_108(__global float* restrict  X_T321, __global const float* restrict  in1, __global const float* restrict  X_T318)
{
  int tid = get_local_id(0);
  float agg[6] = {0, 0, 0, 0, 0, 0, };
  __local float in1_shared[6804];
  int x1_gid = (get_group_id(0) * 4);
  int x0_gid = (get_group_id(1) * 4);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((k1_gid * 84) + (x1_gid * 168)) + (k0_gid * 7140)) + (x0_gid * 14280));
        int c_k1_x1_tid = (tid % 256);
        for (int c_k1_x1_lid = 0; c_k1_x1_lid < 3; c_k1_x1_lid += 1)
        {
          int c_k1_x1_cond = ((c_k1_x1_lid < 2) || (c_k1_x1_tid < 244));
          if (c_k1_x1_cond)
          {
            int c_k1_x1 = ((256 * c_k1_x1_lid) + c_k1_x1_tid);
            for (int k0_x0_lid = 0; k0_x0_lid < 9; k0_x0_lid += 1)
            {
              int lidx = ((9 * c_k1_x1) + k0_x0_lid);
              int gidx = ((gbase + c_k1_x1) + (7140 * k0_x0_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)606899)];
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
      {
        for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
        {
          int c_tid = (tid % 32);
          int x1_tid = ((tid / 32) % 4);
          int x0_tid = ((tid / 128) % 2);
          for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
          {
            int x0 = ((2 * x0_lid) + x0_tid);
            for (int c_lid = 0; c_lid < 3; c_lid += 1)
            {
              int c_cond = ((c_lid < 2) || (c_tid < 20));
              int c = select((int)0, (int)((32 * c_lid) + c_tid), (int)c_cond);
              float val1 = in1_shared[(((((9 * c) + (756 * k1_lid)) + (1512 * x1_tid)) + k0_lid) + (2 * x0))];
              int agg_idx = (c_lid + (x0_lid * 3));
              float agg_rhs = (agg[agg_idx] + val1);
              agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)c_cond);
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  int x1_cond = ((x1_gid != 40) || (x1_tid < 2));
  if (x1_cond)
  {
    for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
    {
      int x0_cond = ((x0_lid < 1) || (x0_gid != 40));
      if (x0_cond)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        for (int c_lid = 0; c_lid < 3; c_lid += 1)
        {
          int c_cond = ((c_lid < 2) || (c_tid < 20));
          if (c_cond)
          {
            int c = ((32 * c_lid) + c_tid);
            float LX_T320 = agg[(c_lid + (x0_lid * 3))];
            int gout_idx = ((c + (3528 * (x0_gid + x0))) + (84 * (x1_gid + x1_tid)));
            float LX_T318 = X_T318[gout_idx];
            float LX_T321 = (LX_T318 / LX_T320);
            X_T321[gout_idx] = LX_T321;
          }
        }
      }
    }
  }
}
