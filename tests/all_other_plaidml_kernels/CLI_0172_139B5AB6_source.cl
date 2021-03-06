#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 2 1
// lid: 256 1 1
// Original:
// X_T812[n, x0, x1, c : _T1253, _T1254, _T1255, _T1256] = +(X_T699[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T812[n, x0, x1, c : _T1253, _T1254, _T1255, _T1256] = +(X_T699[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= -1 + k0 + x0 < 28, 0 <= -1 + k1 + x1 < 28, 0 <= c < 44, 0 <= c < 44, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= -1 + k0 + x0 < 28, 0 <= -1 + k1 + x1 < 28, 0 <= c < 44 }
// Defracted:
// X_T812[n, x0, x1, c : _T1253, _T1254, _T1255, _T1256] = +(X_T699[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T812    X_T699  
//        c        44         1         1  
//       k0         3         0      1232  
//       k1         3         0        44  
//       x0        28      1232      1232  
//       x1        28        44        44  
//      off                   0     -1276  
//      vec                   1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 28
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 28
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 44, 3, 3, 28, 28 }
// Out stride: { 1, 0, 0, 1232, 44 }
// Input 1 offset: -1276
// Input 1 stride: { 1, 1232, 44, 1232, 44 }
// Elementwise input X_T648 shape: fp32(1, 28, 28, 44):(34496, 1232, 44, 1):134.75 KiB
// Elementwise input X_T468 shape: fp32(1, 28, 28, 44):(34496, 1232, 44, 1):134.75 KiB
// Elementwise op: [[pid(normal_A_block_1)]] X_T813 = div(X_T812, X_T648)
// Elementwise op: [[pid(Add)]] X_T814 = add(X_T813, X_T468)
// Tile size: { 44, 3, 3, 16, 4 }
// Contraction output var shape: fp32(1, 28, 28, 44):(34496, 1232, 44, 1):134.75 KiB
// Computed true ops: 1241856
// Computed work groups: 14
// Computed inner loops: 1
// Computed shared mem: 19080
// Computed out regs: 16384
// Computed mem read: 19968
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 2, 1
__kernel void kernel_c42_sdk_296(__global float* restrict  X_T814, __global const float* restrict  in1, __global const float* restrict  X_T648, __global const float* restrict  X_T468)
{
  in1 = (in1 + -1276);
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[4770];
  int x1_gid = (get_group_id(0) * 4);
  int x0_gid = (get_group_id(1) * 16);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((k1_gid * 44) + (x1_gid * 44)) + (k0_gid * 1232)) + (x0_gid * 1232));
        int c_k1_x1_tid = (tid % 256);
        for (int c_k1_x1_lid = 0; c_k1_x1_lid < 2; c_k1_x1_lid += 1)
        {
          int c_k1_x1_cond = ((c_k1_x1_lid < 1) || (c_k1_x1_tid < 8));
          if (c_k1_x1_cond)
          {
            int c_k1_x1 = ((256 * c_k1_x1_lid) + c_k1_x1_tid);
            for (int k0_x0_lid = 0; k0_x0_lid < 18; k0_x0_lid += 1)
            {
              int lidx = (c_k1_x1 + (265 * k0_x0_lid));
              int gidx = ((gbase + c_k1_x1) + (1232 * k0_x0_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)1276, (int)35771)];
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if (((((((-1 * k0_gid) + (-1 * x0_gid)) <= -1) && ((((k0_gid + 3) - 1) + ((x0_gid + 16) - 1)) <= 28)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -1)) && ((((k1_gid + 3) - 1) + ((x1_gid + 4) - 1)) <= 28)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int c_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 4);
            int x0_tid = ((tid / 128) % 2);
            for (int x0_lid = 0; x0_lid < 8; x0_lid += 1)
            {
              int x0 = ((2 * x0_lid) + x0_tid);
              for (int c_lid = 0; c_lid < 2; c_lid += 1)
              {
                int c_cond = ((c_lid < 1) || (c_tid < 12));
                int c = select((int)0, (int)((32 * c_lid) + c_tid), (int)c_cond);
                float val1 = in1_shared[((((c + (44 * k1_lid)) + (44 * x1_tid)) + (265 * k0_lid)) + (265 * x0))];
                int agg_idx = (c_lid + (x0_lid * 2));
                float agg_rhs = (agg[agg_idx] + val1);
                agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)c_cond);
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
            for (int x0_lid = 0; x0_lid < 8; x0_lid += 1)
            {
              int x0 = ((2 * x0_lid) + x0_tid);
              for (int c_lid = 0; c_lid < 2; c_lid += 1)
              {
                int c_cond = ((c_lid < 1) || (c_tid < 12));
                int c = select((int)0, (int)((32 * c_lid) + c_tid), (int)c_cond);
                float val1 = in1_shared[((((c + (44 * k1_lid)) + (44 * x1_tid)) + (265 * k0_lid)) + (265 * x0))];
                int agg_idx = (c_lid + (x0_lid * 2));
                float agg_rhs = (agg[agg_idx] + val1);
                agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)(c_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * (x0_gid + x0))) <= -1) && (((k0_gid + k0_lid) + (x0_gid + x0)) <= 28)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1_tid))) <= -1)) && (((k1_gid + k1_lid) + (x1_gid + x1_tid)) <= 28))));
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
  for (int x0_lid = 0; x0_lid < 8; x0_lid += 1)
  {
    int x0_cond = ((x0_lid < 6) || (x0_gid != 16));
    if (x0_cond)
    {
      int x0 = ((2 * x0_lid) + x0_tid);
      for (int c_lid = 0; c_lid < 2; c_lid += 1)
      {
        int c_cond = ((c_lid < 1) || (c_tid < 12));
        if (c_cond)
        {
          int c = ((32 * c_lid) + c_tid);
          float LX_T812 = agg[(c_lid + (x0_lid * 2))];
          int gout_idx = ((c + (1232 * (x0_gid + x0))) + (44 * (x1_gid + x1_tid)));
          if (((gout_idx >= 0) && (gout_idx < 34496)))
          {
            float LX_T648 = X_T648[gout_idx];
            float LX_T468 = X_T468[gout_idx];
            float LX_T813 = (LX_T812 / LX_T648);
            float LX_T814 = (LX_T813 + LX_T468);
            X_T814[gout_idx] = LX_T814;
          }
        }
      }
    }
  }
}
