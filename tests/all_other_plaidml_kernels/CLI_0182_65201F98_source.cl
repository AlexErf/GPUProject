#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2816 11 1
// lid: 256 1 1
// Original:
// X_T1580[n, x0, x1, g, gco : _T2482, _T2483, _T2484, _T2485, _T2486] = +(X_T1579[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T1566[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T1580[n, x0, x1, g, gco : _T2482, _T2483, _T2484, _T2485, _T2486] = +(X_T1579[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T1566[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 7, 0 <= k1 < 7, 0 <= x0 < 21, 0 <= x1 < 21, 0 <= k0 + 2*x0 < 47, 0 <= k1 + 2*x1 < 47, 0 <= g < 336, 0 <= g + gci < 336, 0 <= g < 336, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 7, 0 <= k1 < 7, 0 <= x0 < 21, 0 <= x1 < 21, 0 <= k0 + 2*x0 < 47, 0 <= k1 + 2*x1 < 47, 0 <= g < 336, 0 <= g + gci < 336 }
// Defracted:
// X_T1580[n, x0, x1, g, gco : _T2482, _T2483, _T2484, _T2485, _T2486] = +(X_T1579[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T1566[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T1580   X_T1579   X_T1566  
//        g       336         1         1         1  
//       k0         7         0     15792      2352  
//       k1         7         0       336       336  
//       x0        21      7056     31584         0  
//       x1        21       336       672         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 336, 7, 7, 21, 21 }
// Out stride: { 1, 0, 0, 7056, 336 }
// Input 1 offset: 0
// Input 1 stride: { 1, 15792, 336, 31584, 672 }
// Input 2 offset: 0
// Input 2 stride: { 1, 2352, 336, 0, 0 }
// Tile size: { 32, 7, 1, 21, 2 }
// Contraction output var shape: fp32(1, 21, 21, 336, 1):(148176, 7056, 336, 1, 1):578.812 KiB
// Computed true ops: 14521248
// Computed work groups: 121
// Computed inner loops: 7
// Computed shared mem: 12936
// Computed out regs: 6144
// Computed mem read: 12928
// Computed mem write: 5376
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2816, 11, 1
__kernel void kernel_c42_sdk_601(__global float* restrict  X_T1580, __global const float* restrict  in1, __global const float* restrict  in2)
{
  int tid = get_local_id(0);
  float agg[6] = {0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3010];
  __local float in2_shared[224];
  int g_gid = (get_group_id(0) * 32);
  int x1_gid = (get_group_id(1) * 2);
  for (int k0_gid = 0; k0_gid < 7; k0_gid += 7)
  {
    for (int k1_gid = 0; k1_gid < 7; k1_gid += 1)
    {
      {
        int gbase = (((g_gid + (k1_gid * 336)) + (x1_gid * 672)) + (k0_gid * 15792));
        int g_tid = (tid % 32);
        int x1_tid = ((tid / 32) % 2);
        int k0_x0_tid = ((tid / 64) % 4);
        for (int k0_x0_lid = 0; k0_x0_lid < 12; k0_x0_lid += 1)
        {
          int k0_x0_cond = ((k0_x0_lid < 11) || (k0_x0_tid < 3));
          if (k0_x0_cond)
          {
            int k0_x0 = ((4 * k0_x0_lid) + k0_x0_tid);
            int lidx = (((47 * g_tid) + (1505 * x1_tid)) + k0_x0);
            int gidx = (((gbase + g_tid) + (672 * x1_tid)) + (15792 * k0_x0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)742223)];
          }
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 336)) + (k0_gid * 2352));
        int g_tid = (tid % 32);
        int k0_tid = ((tid / 32) % 8);
        int k0_cond = (k0_tid < 7);
        if (k0_cond)
        {
          int lidx = ((7 * g_tid) + k0_tid);
          int gidx = ((gbase + g_tid) + (2352 * k0_tid));
          in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)16463)];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k0_lid = 0; k0_lid < 7; k0_lid += 1)
      {
        int g_tid = (tid % 32);
        int x1_tid = ((tid / 32) % 2);
        int x0_tid = ((tid / 64) % 4);
        for (int x0_lid = 0; x0_lid < 6; x0_lid += 1)
        {
          int x0_cond = ((x0_lid < 5) || (x0_tid < 1));
          int x0 = select((int)0, (int)((4 * x0_lid) + x0_tid), (int)x0_cond);
          float val1 = in1_shared[((((47 * g_tid) + (1505 * x1_tid)) + k0_lid) + (2 * x0))];
          float val2 = in2_shared[((7 * g_tid) + k0_lid)];
          float agg_rhs = mad(val2, val1, agg[x0_lid]);
          agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)x0_cond);
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int g_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 2);
  int x0_tid = ((tid / 64) % 4);
  int g_cond = ((g_gid != 320) || (g_tid < 16));
  if (g_cond)
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
          float LX_T1580 = agg[x0_lid];
          int gout_idx = (((g_gid + g_tid) + (7056 * x0)) + (336 * (x1_gid + x1_tid)));
          X_T1580[gout_idx] = LX_T1580;
        }
      }
    }
  }
}
