#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2816 11 1
// lid: 256 1 1
// Original:
// X_T1559[n, x0, x1, c : _T2455, _T2456, _T2457, _T2458] = >(X_T1558[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T1559[n, x0, x1, c : _T2455, _T2456, _T2457, _T2458] = >(X_T1558[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 21, 0 <= x1 < 21, 0 <= k0 + 2*x0 < 43, 0 <= k1 + 2*x1 < 43, 0 <= c < 336, 0 <= c < 336, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 21, 0 <= x1 < 21, 0 <= k0 + 2*x0 < 43, 0 <= k1 + 2*x1 < 43, 0 <= c < 336 }
// Defracted:
// X_T1559[n, x0, x1, c : _T2455, _T2456, _T2457, _T2458] = >(X_T1558[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T1559   X_T1558  
//        c       336         1         1  
//       k0         3         0     14448  
//       k1         3         0       336  
//       x0        21      7056     28896  
//       x1        21       336       672  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 336, 3, 3, 21, 21 }
// Out stride: { 1, 0, 0, 7056, 336 }
// Input 1 offset: 0
// Input 1 stride: { 1, 14448, 336, 28896, 672 }
// Tile size: { 32, 3, 3, 21, 2 }
// Contraction output var shape: fp32(1, 21, 21, 336):(148176, 7056, 336, 1):578.812 KiB
// Computed true ops: 2667168
// Computed work groups: 121
// Computed inner loops: 1
// Computed shared mem: 27520
// Computed out regs: 6144
// Computed mem read: 27520
// Computed mem write: 5376
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2816, 11, 1
__kernel void kernel_c42_sdk_594(__global float* restrict  X_T1559, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[6] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, };
  __local float in1_shared[6880];
  int c_gid = (get_group_id(0) * 32);
  int x1_gid = (get_group_id(1) * 2);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = (((c_gid + (k1_gid * 336)) + (x1_gid * 672)) + (k0_gid * 14448));
        int c_tid = (tid % 32);
        int k0_x0_tid = ((tid / 32) % 8);
        for (int k0_x0_lid = 0; k0_x0_lid < 6; k0_x0_lid += 1)
        {
          int k0_x0_cond = ((k0_x0_lid < 5) || (k0_x0_tid < 3));
          if (k0_x0_cond)
          {
            int k0_x0 = ((8 * k0_x0_lid) + k0_x0_tid);
            for (int k1_x1_lid = 0; k1_x1_lid < 5; k1_x1_lid += 1)
            {
              int lidx = (((215 * c_tid) + k0_x0) + (43 * k1_x1_lid));
              int gidx = (((gbase + c_tid) + (14448 * k0_x0)) + (336 * k1_x1_lid));
              in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)621263)];
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
          for (int x0_lid = 0; x0_lid < 6; x0_lid += 1)
          {
            int x0_cond = ((x0_lid < 5) || (x0_tid < 1));
            int x0 = select((int)0, (int)((4 * x0_lid) + x0_tid), (int)x0_cond);
            float val1 = in1_shared[(((((215 * c_tid) + (43 * k1_lid)) + (86 * x1_tid)) + k0_lid) + (2 * x0))];
            float agg_rhs = select((float)agg[x0_lid], (float)val1, (int)(val1 > agg[x0_lid]));
            agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)x0_cond);
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 2);
  int x0_tid = ((tid / 64) % 4);
  int c_cond = ((c_gid != 320) || (c_tid < 16));
  if (c_cond)
  {
    int x1_cond = ((x1_gid != 20) || (x1_tid < 1));
    if (x1_cond)
    {
      for (int x0_lid = 0; x0_lid < 6; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 5) || (x0_tid < 1));
        if (x0_cond)
        {
          int x0 = ((4 * x0_lid) + x0_tid);
          float LX_T1559 = agg[x0_lid];
          LX_T1559 = select((float)LX_T1559, (float)0, (int)(LX_T1559 == (float)-FLT_MAX));
          int gout_idx = (((c_gid + c_tid) + (7056 * x0)) + (336 * (x1_gid + x1_tid)));
          X_T1559[gout_idx] = LX_T1559;
        }
      }
    }
  }
}
