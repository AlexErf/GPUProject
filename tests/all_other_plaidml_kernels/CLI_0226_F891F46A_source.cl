#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 7 1
// lid: 256 1 1
// Original:
// X_T1745[n, x0, x1, c : _T2758, _T2759, _T2760, _T2761] = +(X_T1406[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T1745[n, x0, x1, c : _T2758, _T2759, _T2760, _T2761] = +(X_T1406[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= -1 + k0 + x0 < 14, 0 <= -1 + k1 + x1 < 14, 0 <= c < 88, 0 <= c < 88, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= -1 + k0 + x0 < 14, 0 <= -1 + k1 + x1 < 14, 0 <= c < 88 }
// Defracted:
// X_T1745[n, x0, x1, c : _T2758, _T2759, _T2760, _T2761] = +(X_T1406[n, -1 + k0 + x0, -1 + k1 + x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T1745   X_T1406  
//        c        88         1         1  
//       k0         3         0      1232  
//       k1         3         0        88  
//       x0        14      1232      1232  
//       x1        14        88        88  
//      off                   0     -1320  
//      vec                   1         1  
// Constraint: (0,-1,0,-1,0) <= -1
// Constraint: (0,1,0,1,0) <= 14
// Constraint: (0,0,-1,0,-1) <= -1
// Constraint: (0,0,1,0,1) <= 14
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 88, 3, 3, 14, 14 }
// Out stride: { 1, 0, 0, 1232, 88 }
// Input 1 offset: -1320
// Input 1 stride: { 1, 1232, 88, 1232, 88 }
// Elementwise input X_T1360 shape: fp32(1, 14, 14, 88):(17248, 1232, 88, 1):67.375 KiB
// Elementwise op: [[pid(normal_A_block_6)]] X_T1746 = div(X_T1745, X_T1360)
// Elementwise op: [[pid(Add)]] X_T1747 = add(X_T1746, X_T1746)
// Tile size: { 32, 3, 3, 14, 2 }
// Contraction output var shape: fp32(1, 14, 14, 88):(17248, 1232, 88, 1):67.375 KiB
// Computed true ops: 620928
// Computed work groups: 21
// Computed inner loops: 1
// Computed shared mem: 8464
// Computed out regs: 4096
// Computed mem read: 8304
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 7, 1
__kernel void kernel_c42_sdk_663(__global float* restrict  X_T1747, __global const float* restrict  in1, __global const float* restrict  X_T1360)
{
  in1 = (in1 + -1320);
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[2116];
  int c_gid = (get_group_id(0) * 32);
  int x1_gid = (get_group_id(1) * 2);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = (((c_gid + (k1_gid * 88)) + (x1_gid * 88)) + (k0_gid * 1232));
        int c_tid = (tid % 32);
        int k0_x0_tid = ((tid / 32) % 8);
        for (int k0_x0_lid = 0; k0_x0_lid < 2; k0_x0_lid += 1)
        {
          int k0_x0 = ((8 * k0_x0_lid) + k0_x0_tid);
          for (int k1_x1_lid = 0; k1_x1_lid < 4; k1_x1_lid += 1)
          {
            int lidx = ((c_tid + (33 * k0_x0)) + (529 * k1_x1_lid));
            int gidx = (((gbase + c_tid) + (1232 * k0_x0)) + (88 * k1_x1_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)1320, (int)18567)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if ((((((-1 * k0_gid) <= -1) && ((((k0_gid + 3) - 1) + 13) <= 14)) && (((-1 * k1_gid) + (-1 * x1_gid)) <= -1)) && ((((k1_gid + 3) - 1) + ((x1_gid + 2) - 1)) <= 14)))
      {
        for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
        {
          for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
          {
            int c_tid = (tid % 32);
            int x1_tid = ((tid / 32) % 2);
            int x0_tid = ((tid / 64) % 4);
            for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
            {
              int x0_cond = ((x0_lid < 3) || (x0_tid < 2));
              int x0 = select((int)0, (int)((4 * x0_lid) + x0_tid), (int)x0_cond);
              float val1 = in1_shared[((((c_tid + (529 * k1_lid)) + (529 * x1_tid)) + (33 * k0_lid)) + (33 * x0))];
              float agg_rhs = (agg[x0_lid] + val1);
              agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)x0_cond);
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
            int x1_tid = ((tid / 32) % 2);
            int x0_tid = ((tid / 64) % 4);
            for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
            {
              int x0_cond = ((x0_lid < 3) || (x0_tid < 2));
              int x0 = select((int)0, (int)((4 * x0_lid) + x0_tid), (int)x0_cond);
              float val1 = in1_shared[((((c_tid + (529 * k1_lid)) + (529 * x1_tid)) + (33 * k0_lid)) + (33 * x0))];
              float agg_rhs = (agg[x0_lid] + val1);
              agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)(x0_cond && ((((((-1 * (k0_gid + k0_lid)) + (-1 * x0)) <= -1) && (((k0_gid + k0_lid) + x0) <= 14)) && (((-1 * (k1_gid + k1_lid)) + (-1 * (x1_gid + x1_tid))) <= -1)) && (((k1_gid + k1_lid) + (x1_gid + x1_tid)) <= 14))));
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 2);
  int x0_tid = ((tid / 64) % 4);
  int c_cond = ((c_gid != 64) || (c_tid < 24));
  if (c_cond)
  {
    for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
    {
      int x0_cond = ((x0_lid < 3) || (x0_tid < 2));
      if (x0_cond)
      {
        int x0 = ((4 * x0_lid) + x0_tid);
        float LX_T1745 = agg[x0_lid];
        int gout_idx = (((c_gid + c_tid) + (1232 * x0)) + (88 * (x1_gid + x1_tid)));
        if (((gout_idx >= 0) && (gout_idx < 17248)))
        {
          float LX_T1360 = X_T1360[gout_idx];
          float LX_T1746 = (LX_T1745 / LX_T1360);
          float LX_T1747 = (LX_T1746 + LX_T1746);
          X_T1747[gout_idx] = LX_T1747;
        }
      }
    }
  }
}
