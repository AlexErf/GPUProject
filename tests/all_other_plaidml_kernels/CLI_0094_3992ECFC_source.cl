#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2816 11 1
// lid: 256 1 1
// Original:
// X_T203[n, x0, x1, c : _T262, _T263, _T264, _T265] = +(X_T202[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T203[n, x0, x1, c : _T262, _T263, _T264, _T265] = +(X_T202[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= c < 42, 0 <= c < 42, 0 <= x0 < 83, 0 <= x1 < 83, 0 <= -1 + k0 + x0 < 83, 0 <= -1 + k1 + x1 < 83, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= c < 42, 0 <= x0 < 83, 0 <= x1 < 83, 0 <= -1 + k0 + x0 < 83, 0 <= -1 + k1 + x1 < 83 }
// Defracted:
// X_T203[n, x0, x1, c : _T262, _T263, _T264, _T265] = +(X_T202[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T203    X_T202  
//        c        42         1         1  
//       k0         3         0      3486  
//       k1         3         0        42  
//       x0        83      3486      3486  
//       x1        83        42        42  
//      off                   0     -3528  
//      vec                   1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 83
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 83
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 42, 3, 3, 83, 83 }
// Out stride: { 1, 0, 0, 3486, 42 }
// Input 1 offset: -3528
// Input 1 stride: { 1, 3486, 42, 3486, 42 }
// Elementwise input X_T201 shape: fp32(1, 83, 83, 42):(289338, 3486, 42, 1):1130.23 KiB
// Elementwise input X_T97 shape: fp32(1, 83, 83, 42):(289338, 3486, 42, 1):1130.23 KiB
// Elementwise op: [[pid(reduction_A_block_stem_1)]] X_T204 = div(X_T201, X_T203)
// Elementwise op: [[pid(Add)]] X_T205 = add(X_T97, X_T204)
// Tile size: { 42, 3, 3, 8, 8 }
// Contraction output var shape: fp32(1, 83, 83, 42):(289338, 3486, 42, 1):1130.23 KiB
// Computed true ops: 10416168
// Computed work groups: 121
// Computed inner loops: 1
// Computed shared mem: 16840
// Computed out regs: 16384
// Computed mem read: 17792
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2816, 11, 1
__kernel void kernel_c42_sdk_59(__global float* restrict  X_T205, __global const float* restrict  in1, __global const float* restrict  X_T201, __global const float* restrict  X_T97)
{
  in1 = (in1 + -3528);
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[4210];
  int x1_gid = (get_group_id(0) * 8);
  int x0_gid = (get_group_id(1) * 8);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((k1_gid * 42) + (x1_gid * 42)) + (k0_gid * 3486)) + (x0_gid * 3486));
        int c_k1_x1_tid = (tid % 256);
        for (int c_k1_x1_lid = 0; c_k1_x1_lid < 2; c_k1_x1_lid += 1)
        {
          int c_k1_x1_cond = ((c_k1_x1_lid < 1) || (c_k1_x1_tid < 164));
          if (c_k1_x1_cond)
          {
            int c_k1_x1 = ((256 * c_k1_x1_lid) + c_k1_x1_tid);
            for (int k0_x0_lid = 0; k0_x0_lid < 10; k0_x0_lid += 1)
            {
              int lidx = (c_k1_x1 + (421 * k0_x0_lid));
              int gidx = ((gbase + c_k1_x1) + (3486 * k0_x0_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)3528, (int)292865)];
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1) && ((((k0_gid + 3) - 1) + ((x0_gid + 8) - 1)) <= 83)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -1)) && ((((k1_gid + 3) - 1) + ((x1_gid + 8) - 1)) <= 83)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int c_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
            {
              int x1 = ((4 * x1_lid) + x1_tid);
              for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
              {
                int x0 = ((2 * x0_lid) + x0_tid);
                for (int c_lid = 0; c_lid < 2; c_lid += 1)
                {
                  int c_cond = ((c_lid < 1) || (c_tid < 10));
                  int c = select((int)0, (int)((32 * c_lid) + c_tid), (int)c_cond);
                  float val1 = in1_shared[((((c + (42 * k1_lid)) + (42 * x1)) + (421 * k0_lid)) + (421 * x0))];
                  int agg_idx = ((c_lid + (x1_lid * 2)) + (x0_lid * 4));
                  float agg_rhs = (agg[agg_idx] + val1);
                  agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)c_cond);
                }
              }
            }
          }
        }
      }
      else
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int c_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
            {
              int x1 = ((4 * x1_lid) + x1_tid);
              for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
              {
                int x0 = ((2 * x0_lid) + x0_tid);
                for (int c_lid = 0; c_lid < 2; c_lid += 1)
                {
                  int c_cond = ((c_lid < 1) || (c_tid < 10));
                  int c = select((int)0, (int)((32 * c_lid) + c_tid), (int)c_cond);
                  float val1 = in1_shared[((((c + (42 * k1_lid)) + (42 * x1)) + (421 * k0_lid)) + (421 * x0))];
                  int agg_idx = ((c_lid + (x1_lid * 2)) + (x0_lid * 4));
                  float agg_rhs = (agg[agg_idx] + val1);
                  agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)(c_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0))) <= -1) && (((k0_gid + k0_lid) + (x0_gid + x0)) <= 83)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1))) <= -1)) && (((k1_gid + k1_lid) + (x1_gid + x1)) <= 83))));
                }
              }
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
  for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
  {
    int x1_cond = (((x1_lid < 0) || ((x1_gid != 80) || (x1_tid < 3))) && ((x1_lid < 1) || (x1_gid != 80)));
    if (x1_cond)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
      {
        int x0_cond = (((x0_lid < 1) || ((x0_gid != 80) || (x0_tid < 1))) && ((x0_lid < 2) || (x0_gid != 80)));
        if (x0_cond)
        {
          int x0 = ((2 * x0_lid) + x0_tid);
          for (int c_lid = 0; c_lid < 2; c_lid += 1)
          {
            int c_cond = ((c_lid < 1) || (c_tid < 10));
            if (c_cond)
            {
              int c = ((32 * c_lid) + c_tid);
              float LX_T203 = agg[((c_lid + (x1_lid * 2)) + (x0_lid * 4))];
              int gout_idx = ((c + (3486 * (x0_gid + x0))) + (42 * (x1_gid + x1)));
              if (((gout_idx >= 0) && (gout_idx < 289338)))
              {
                float LX_T201 = X_T201[gout_idx];
                float LX_T97 = X_T97[gout_idx];
                float LX_T204 = (LX_T201 / LX_T203);
                float LX_T205 = (LX_T97 + LX_T204);
                X_T205[gout_idx] = LX_T205;
              }
            }
          }
        }
      }
    }
  }
}
