#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 11 1
// lid: 256 1 1
// Original:
// X_T3197[n, x0, x1, c : _T5101, _T5102, _T5103, _T5104] = +(X_T3196[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T3197[n, x0, x1, c : _T5101, _T5102, _T5103, _T5104] = +(X_T3196[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 11, 0 <= x1 < 11, 0 <= -1 + k0 + x0 < 11, 0 <= -1 + k1 + x1 < 11, 0 <= c < 672, 0 <= c < 672, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 11, 0 <= x1 < 11, 0 <= -1 + k0 + x0 < 11, 0 <= -1 + k1 + x1 < 11, 0 <= c < 672 }
// Defracted:
// X_T3197[n, x0, x1, c : _T5101, _T5102, _T5103, _T5104] = +(X_T3196[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T3197   X_T3196  
//        c       672         1         1  
//       k0         3         0      7392  
//       k1         3         0       672  
//       x0        11      7392      7392  
//       x1        11       672       672  
//      off                   0     -8064  
//      vec                   1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 11
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 11
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 672, 3, 3, 11, 11 }
// Out stride: { 1, 0, 0, 7392, 672 }
// Input 1 offset: -8064
// Input 1 stride: { 1, 7392, 672, 7392, 672 }
// Elementwise input X_T3195 shape: fp32(1, 11, 11, 672):(81312, 7392, 672, 1):317.625 KiB
// Elementwise op: [[pid(normal_A_block_13)]] X_T3198 = div(X_T3195, X_T3197)
// Elementwise op: [[pid(Add)]] X_T3199 = add(X_T3198, X_T3198)
// Tile size: { 256, 3, 1, 11, 1 }
// Contraction output var shape: fp32(1, 11, 11, 672):(81312, 7392, 672, 1):317.625 KiB
// Computed true ops: 2927232
// Computed work groups: 33
// Computed inner loops: 3
// Computed shared mem: 13312
// Computed out regs: 16384
// Computed mem read: 13664
// Computed mem write: 11264
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 11, 1
__kernel void kernel_c42_sdk_1240(__global float* restrict  X_T3199, __global const float* restrict  in1, __global const float* restrict  X_T3195)
{
  in1 = (in1 + -8064);
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3328];
  int c_gid = (get_group_id(0) * 256);
  int x1_gid = get_group_id(1);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 1)
    {
      {
        int gbase = (((c_gid + (k1_gid * 672)) + (x1_gid * 672)) + (k0_gid * 7392));
        int c_tid = (tid % 256);
        for (int k0_x0_lid = 0; k0_x0_lid < 13; k0_x0_lid += 1)
        {
          int lidx = ((13 * c_tid) + k0_x0_lid);
          int gidx = ((gbase + c_tid) + (7392 * k0_x0_lid));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)8064, (int)89375)];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if ((((((-1 * k0_gid) <= -1) && ((((k0_gid + 3) - 1) + 10) <= 11)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -1)) && ((((k1_gid + 1) - 1) + ((x1_gid + 1) - 1)) <= 11)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          int c_tid = (tid % 32);
          int x0_tid = ((tid / 32) % 8);
          for (int c_lid = 0; c_lid < 8; c_lid += 1)
          {
            int c = ((32 * c_lid) + c_tid);
            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
            {
              int x0_cond = ((x0_lid < 1) || (x0_tid < 3));
              int x0 = select((int)0, (int)((8 * x0_lid) + x0_tid), (int)x0_cond);
              float val1 = in1_shared[(((13 * c) + k0_lid) + x0)];
              int agg_idx = (c_lid + (x0_lid * 8));
              float agg_rhs = (agg[agg_idx] + val1);
              agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)x0_cond);
            }
          }
        }
      }
      else
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          int c_tid = (tid % 32);
          int x0_tid = ((tid / 32) % 8);
          for (int c_lid = 0; c_lid < 8; c_lid += 1)
          {
            int c = ((32 * c_lid) + c_tid);
            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
            {
              int x0_cond = ((x0_lid < 1) || (x0_tid < 3));
              int x0 = select((int)0, (int)((8 * x0_lid) + x0_tid), (int)x0_cond);
              float val1 = in1_shared[(((13 * c) + k0_lid) + x0)];
              int agg_idx = (c_lid + (x0_lid * 8));
              float agg_rhs = (agg[agg_idx] + val1);
              agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)(x0_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * x0)) <= -1) && (((k0_gid + k0_lid) + x0) <= 11)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -1)) && ((k1_gid + x1_gid) <= 11))));
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 32);
  int x0_tid = ((tid / 32) % 8);
  for (int c_lid = 0; c_lid < 8; c_lid += 1)
  {
    int c_cond = ((c_lid < 5) || (c_gid != 512));
    if (c_cond)
    {
      int c = ((32 * c_lid) + c_tid);
      for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 1) || (x0_tid < 3));
        if (x0_cond)
        {
          int x0 = ((8 * x0_lid) + x0_tid);
          float LX_T3197 = agg[(c_lid + (x0_lid * 8))];
          int gout_idx = (((c_gid + c) + (7392 * x0)) + (672 * x1_gid));
          if (((gout_idx >= 0) && (gout_idx < 81312)))
          {
            float LX_T3195 = X_T3195[gout_idx];
            float LX_T3198 = (LX_T3195 / LX_T3197);
            float LX_T3199 = (LX_T3198 + LX_T3198);
            X_T3199[gout_idx] = LX_T3199;
          }
        }
      }
    }
  }
}
