#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 7 1
// lid: 256 1 1
// Original:
// X_T1259[n, x0, x1, c : _T1969, _T1970, _T1971, _T1972] = +(X_T1211[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T1259[n, x0, x1, c : _T1969, _T1970, _T1971, _T1972] = +(X_T1211[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + 2*x0 < 29, 0 <= k1 + 2*x1 < 29, 0 <= c < 88, 0 <= c < 88, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + 2*x0 < 29, 0 <= k1 + 2*x1 < 29, 0 <= c < 88 }
// Defracted:
// X_T1259[n, x0, x1, c : _T1969, _T1970, _T1971, _T1972] = +(X_T1211[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T1259   X_T1211  
//        c        88         1         1  
//       k0         3         0      2552  
//       k1         3         0        88  
//       x0        14      1232      5104  
//       x1        14        88       176  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 88, 3, 3, 14, 14 }
// Out stride: { 1, 0, 0, 1232, 88 }
// Input 1 offset: 0
// Input 1 stride: { 1, 2552, 88, 5104, 176 }
// Tile size: { 32, 3, 3, 2, 14 }
// Contraction output var shape: fp32(1, 14, 14, 88):(17248, 1232, 88, 1):67.375 KiB
// Computed true ops: 310464
// Computed work groups: 21
// Computed inner loops: 1
// Computed shared mem: 18560
// Computed out regs: 4096
// Computed mem read: 18560
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 7, 1
__kernel void kernel_c42_sdk_472(__global float* restrict  X_T1259, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[4640];
  int c_gid = (get_group_id(0) * 32);
  int x0_gid = (get_group_id(1) * 2);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = (((c_gid + (k1_gid * 88)) + (k0_gid * 2552)) + (x0_gid * 5104));
        int c_tid = (tid % 32);
        int k1_x1_k0_x0_tid = ((tid / 32) % 8);
        for (int k1_x1_k0_x0_lid = 0; k1_x1_k0_x0_lid < 19; k1_x1_k0_x0_lid += 1)
        {
          int k1_x1_k0_x0_cond = ((k1_x1_k0_x0_lid < 18) || (k1_x1_k0_x0_tid < 1));
          if (k1_x1_k0_x0_cond)
          {
            int k1_x1_k0_x0 = ((8 * k1_x1_k0_x0_lid) + k1_x1_k0_x0_tid);
            int lidx = ((145 * c_tid) + k1_x1_k0_x0);
            int gidx = ((gbase + c_tid) + (88 * k1_x1_k0_x0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)74007)];
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
          for (int x1_lid = 0; x1_lid < 4; x1_lid += 1)
          {
            int x1_cond = ((x1_lid < 3) || (x1_tid < 2));
            int x1 = select((int)0, (int)((4 * x1_lid) + x1_tid), (int)x1_cond);
            float val1 = in1_shared[(((((145 * c_tid) + k1_lid) + (2 * x1)) + (29 * k0_lid)) + (58 * x0_tid))];
            float agg_rhs = (agg[x1_lid] + val1);
            agg[x1_lid] = select((float)agg[x1_lid], (float)agg_rhs, (int)x1_cond);
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  int c_cond = ((c_gid != 64) || (c_tid < 24));
  if (c_cond)
  {
    for (int x1_lid = 0; x1_lid < 4; x1_lid += 1)
    {
      int x1_cond = ((x1_lid < 3) || (x1_tid < 2));
      if (x1_cond)
      {
        int x1 = ((4 * x1_lid) + x1_tid);
        float LX_T1259 = agg[x1_lid];
        int gout_idx = (((c_gid + c_tid) + (1232 * (x0_gid + x0_tid))) + (88 * x1));
        X_T1259[gout_idx] = LX_T1259;
      }
    }
  }
}
