#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4352 4 1
// lid: 256 1 1
// Original:
// X_T1998[n, x0, x1, c : _T2797, _T2798, _T2799, _T2800] = >(X_T1918[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T1998[n, x0, x1, c : _T2797, _T2798, _T2799, _T2800] = >(X_T1918[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 8, 0 <= x1 < 8, 0 <= k0 + 2*x0 < 17, 0 <= k1 + 2*x1 < 17, 0 <= c < 1088, 0 <= c < 1088, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 8, 0 <= x1 < 8, 0 <= k0 + 2*x0 < 17, 0 <= k1 + 2*x1 < 17, 0 <= c < 1088 }
// Defracted:
// X_T1998[n, x0, x1, c : _T2797, _T2798, _T2799, _T2800] = >(X_T1918[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T1998   X_T1918  
//        c      1088         1         1  
//       k0         3         0     18496  
//       k1         3         0      1088  
//       x0         8      8704     36992  
//       x1         8      1088      2176  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 1088, 3, 3, 8, 8 }
// Out stride: { 1, 0, 0, 8704, 1088 }
// Input 1 offset: 0
// Input 1 stride: { 1, 18496, 1088, 36992, 2176 }
// Tile size: { 64, 3, 3, 8, 2 }
// Contraction output var shape: fp32(1, 8, 8, 1088):(69632, 8704, 1088, 1):272 KiB
// Computed true ops: 1253376
// Computed work groups: 68
// Computed inner loops: 1
// Computed shared mem: 21760
// Computed out regs: 4096
// Computed mem read: 21760
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4352, 4, 1
__kernel void kernel_c51_sdk_652(__global float* restrict  X_T1998, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, };
  __local float in1_shared[5440];
  int c_gid = (get_group_id(0) * 64);
  int x1_gid = (get_group_id(1) * 2);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = (((c_gid + (k1_gid * 1088)) + (x1_gid * 2176)) + (k0_gid * 18496));
        int c_tid = (tid % 64);
        int k1_x1_tid = ((tid / 64) % 4);
        for (int k1_x1_lid = 0; k1_x1_lid < 2; k1_x1_lid += 1)
        {
          int k1_x1_cond = ((k1_x1_lid < 1) || (k1_x1_tid < 1));
          if (k1_x1_cond)
          {
            int k1_x1 = ((4 * k1_x1_lid) + k1_x1_tid);
            for (int k0_x0_lid = 0; k0_x0_lid < 17; k0_x0_lid += 1)
            {
              int lidx = (((85 * c_tid) + (17 * k1_x1)) + k0_x0_lid);
              int gidx = (((gbase + c_tid) + (1088 * k1_x1)) + (18496 * k0_x0_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)314431)];
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
          int x1_tid = ((tid / 32) % 2);
          int x0_tid = ((tid / 64) % 4);
          for (int c_lid = 0; c_lid < 2; c_lid += 1)
          {
            int c = ((32 * c_lid) + c_tid);
            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
            {
              int x0 = ((4 * x0_lid) + x0_tid);
              float val1 = in1_shared[(((((85 * c) + (17 * k1_lid)) + (34 * x1_tid)) + k0_lid) + (2 * x0))];
              int agg_idx = (c_lid + (x0_lid * 2));
              float agg_rhs = select((float)agg[agg_idx], (float)val1, (int)(val1 > agg[agg_idx]));
              agg[agg_idx] = agg_rhs;
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
  for (int c_lid = 0; c_lid < 2; c_lid += 1)
  {
    int c = ((32 * c_lid) + c_tid);
    for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
    {
      int x0 = ((4 * x0_lid) + x0_tid);
      float LX_T1998 = agg[(c_lid + (x0_lid * 2))];
      LX_T1998 = select((float)LX_T1998, (float)0, (int)(LX_T1998 == (float)-FLT_MAX));
      int gout_idx = (((c_gid + c) + (8704 * x0)) + (1088 * (x1_gid + x1_tid)));
      X_T1998[gout_idx] = LX_T1998;
    }
  }
}
