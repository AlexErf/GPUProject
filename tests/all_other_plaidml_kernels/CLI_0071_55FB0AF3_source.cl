#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 21 1
// lid: 256 1 1
// Original:
// X_T60[n, x0, x1, c : _T40, _T41, _T42, _T43] = >(X_T59[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T60[n, x0, x1, c : _T40, _T41, _T42, _T43] = >(X_T59[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= c < 42, 0 <= c < 42, 0 <= x0 < 83, 0 <= x1 < 83, 0 <= k0 + 2*x0 < 167, 0 <= k1 + 2*x1 < 167, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= c < 42, 0 <= x0 < 83, 0 <= x1 < 83, 0 <= k0 + 2*x0 < 167, 0 <= k1 + 2*x1 < 167 }
// Defracted:
// X_T60[n, x0, x1, c : _T40, _T41, _T42, _T43] = >(X_T59[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T60     X_T59  
//        c        42         1         1  
//       k0         3         0      7014  
//       k1         3         0        42  
//       x0        83      3486     14028  
//       x1        83        42        84  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 42, 3, 3, 83, 83 }
// Out stride: { 1, 0, 0, 3486, 42 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7014, 42, 14028, 84 }
// Tile size: { 42, 3, 3, 4, 4 }
// Contraction output var shape: fp32(1, 83, 83, 42):(289338, 3486, 42, 1):1130.23 KiB
// Computed true ops: 5208084
// Computed work groups: 441
// Computed inner loops: 1
// Computed shared mem: 13608
// Computed out regs: 4096
// Computed mem read: 13568
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 21, 1
__kernel void kernel_c42_sdk_7(__global float* restrict  X_T60, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, };
  __local float in1_shared[3402];
  int x1_gid = (get_group_id(0) * 4);
  int x0_gid = (get_group_id(1) * 4);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((k1_gid * 42) + (x1_gid * 84)) + (k0_gid * 7014)) + (x0_gid * 14028));
        int c_k1_x1_tid = (tid % 256);
        for (int c_k1_x1_lid = 0; c_k1_x1_lid < 2; c_k1_x1_lid += 1)
        {
          int c_k1_x1_cond = ((c_k1_x1_lid < 1) || (c_k1_x1_tid < 122));
          if (c_k1_x1_cond)
          {
            int c_k1_x1 = ((256 * c_k1_x1_lid) + c_k1_x1_tid);
            for (int k0_x0_lid = 0; k0_x0_lid < 9; k0_x0_lid += 1)
            {
              int lidx = ((9 * c_k1_x1) + k0_x0_lid);
              int gidx = ((gbase + c_k1_x1) + (7014 * k0_x0_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)1171337)];
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
            for (int c_lid = 0; c_lid < 2; c_lid += 1)
            {
              int c_cond = ((c_lid < 1) || (c_tid < 10));
              int c = select((int)0, (int)((32 * c_lid) + c_tid), (int)c_cond);
              float val1 = in1_shared[(((((9 * c) + (378 * k1_lid)) + (756 * x1_tid)) + k0_lid) + (2 * x0))];
              int agg_idx = (c_lid + (x0_lid * 2));
              float agg_rhs = select((float)agg[agg_idx], (float)val1, (int)(val1 > agg[agg_idx]));
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
  int x1_cond = ((x1_gid != 80) || (x1_tid < 3));
  if (x1_cond)
  {
    for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
    {
      int x0_cond = ((x0_lid < 1) || ((x0_gid != 80) || (x0_tid < 1)));
      if (x0_cond)
      {
        int x0 = ((2 * x0_lid) + x0_tid);
        for (int c_lid = 0; c_lid < 2; c_lid += 1)
        {
          int c_cond = ((c_lid < 1) || (c_tid < 10));
          if (c_cond)
          {
            int c = ((32 * c_lid) + c_tid);
            float LX_T60 = agg[(c_lid + (x0_lid * 2))];
            LX_T60 = select((float)LX_T60, (float)0, (int)(LX_T60 == (float)-FLT_MAX));
            int gout_idx = ((c + (3486 * (x0_gid + x0))) + (42 * (x1_gid + x1_tid)));
            X_T60[gout_idx] = LX_T60;
          }
        }
      }
    }
  }
}
