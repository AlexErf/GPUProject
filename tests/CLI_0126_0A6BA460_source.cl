#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 4 1
// lid: 256 1 1
// Original:
// X_T328[n, x0, x1, g, gco : _T490, _T491, _T492, _T493, _T494] = +(X_T327[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T325[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T328[n, x0, x1, g, gco : _T490, _T491, _T492, _T493, _T494] = +(X_T327[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T325[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= g < 22, 0 <= g + gci < 22, 0 <= g < 22, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + 2*x0 < 59, 0 <= k1 + 2*x1 < 59, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 5, 0 <= k1 < 5, 0 <= g < 22, 0 <= g + gci < 22, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + 2*x0 < 59, 0 <= k1 + 2*x1 < 59 }
// Defracted:
// X_T328[n, x0, x1, g, gco : _T490, _T491, _T492, _T493, _T494] = +(X_T327[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T325[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T328    X_T327    X_T325  
//        g        22         1         1         1  
//       k0         5         0      1298       110  
//       k1         5         0        22        22  
//       x0        28       616      2596         0  
//       x1        28        22        44         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 22, 5, 5, 28, 28 }
// Out stride: { 1, 0, 0, 616, 22 }
// Input 1 offset: 0
// Input 1 stride: { 1, 1298, 22, 2596, 44 }
// Input 2 offset: 0
// Input 2 stride: { 1, 110, 22, 0, 0 }
// Tile size: { 22, 5, 5, 8, 4 }
// Contraction output var shape: fp32(1, 28, 28, 22, 1):(17248, 616, 22, 1, 1):67.375 KiB
// Computed true ops: 862400
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 20592
// Computed out regs: 4096
// Computed mem read: 20480
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 4, 1
__kernel void kernel_c42_sdk_112(__global float* restrict  X_T328, __global const float* restrict  in1, __global const float* restrict  in2)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[4598];
  __local float in2_shared[550];
  int x1_gid = (get_group_id(0) * 4);
  int x0_gid = (get_group_id(1) * 8);
  for (int k0_gid = 0; k0_gid < 5; k0_gid += 5)
  {
    for (int k1_gid = 0; k1_gid < 5; k1_gid += 5)
    {
      {
        int gbase = ((((k1_gid * 22) + (x1_gid * 44)) + (k0_gid * 1298)) + (x0_gid * 2596));
        int g_k1_x1_tid = (tid % 256);
        int g_k1_x1_cond = (g_k1_x1_tid < 242);
        if (g_k1_x1_cond)
        {
          for (int k0_x0_lid = 0; k0_x0_lid < 19; k0_x0_lid += 1)
          {
            int lidx = ((19 * g_k1_x1_tid) + k0_x0_lid);
            int gidx = ((gbase + g_k1_x1_tid) + (1298 * k0_x0_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)76581)];
          }
        }
      }
      {
        int gbase = ((k1_gid * 22) + (k0_gid * 110));
        int g_k1_k0_tid = (tid % 256);
        for (int g_k1_k0_lid = 0; g_k1_k0_lid < 3; g_k1_k0_lid += 1)
        {
          int g_k1_k0_cond = ((g_k1_k0_lid < 2) || (g_k1_k0_tid < 38));
          if (g_k1_k0_cond)
          {
            int g_k1_k0 = ((256 * g_k1_k0_lid) + g_k1_k0_tid);
            int gidx = (gbase + g_k1_k0);
            in2_shared[g_k1_k0] = in2[clamp((int)gidx, (int)0, (int)549)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k0_lid = 0; k0_lid < 5; k0_lid += 1)
      {
        for (int k1_lid = 0; k1_lid < 5; k1_lid += 1)
        {
          int g_tid = (tid % 32);
          int x1_tid = ((tid / 32) % 4);
          int x0_tid = ((tid / 128) % 2);
          for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
          {
            int x0 = ((2 * x0_lid) + x0_tid);
            int g_cond = (g_tid < 22);
            int g = select((int)0, (int)g_tid, (int)g_cond);
            float val1 = in1_shared[(((((19 * g) + (418 * k1_lid)) + (836 * x1_tid)) + k0_lid) + (2 * x0))];
            float val2 = in2_shared[((g + (22 * k1_lid)) + (110 * k0_lid))];
            float agg_rhs = mad(val2, val1, agg[x0_lid]);
            agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)g_cond);
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int g_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  for (int x0_lid = 0; x0_lid < 4; x0_lid += 1)
  {
    int x0_cond = ((x0_lid < 2) || (x0_gid != 24));
    if (x0_cond)
    {
      int x0 = ((2 * x0_lid) + x0_tid);
      int g_cond = (g_tid < 22);
      if (g_cond)
      {
        float LX_T328 = agg[x0_lid];
        int gout_idx = ((g_tid + (616 * (x0_gid + x0))) + (22 * (x1_gid + x1_tid)));
        X_T328[gout_idx] = LX_T328;
      }
    }
  }
}
