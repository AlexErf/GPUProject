#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 4 1
// lid: 256 1 1
// Original:
// X_T172[n, x0, x1, g, gco : _T162, _T163, _T164, _T165, _T166] = +(X_T171[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T51[k0, k1, g, gco, gci])
// With Index Variables Made Integral:
// X_T172[n, x0, x1, g, gco : _T162, _T163, _T164, _T165, _T166] = +(X_T171[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T51[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + 2*x0 < 57, 0 <= k1 + 2*x1 < 57, 0 <= g < 128, 0 <= g + gci < 128, 0 <= g < 128, 0 <= 500000000 + g < 1000000000, 0 <= 500000000 + gci < 1000000000, 0 <= 500000000 + gco < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= gco < 1, 0 <= gci < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + 2*x0 < 57, 0 <= k1 + 2*x1 < 57, 0 <= g < 128, 0 <= g + gci < 128 }
// Defracted:
// X_T172[n, x0, x1, g, gco : _T162, _T163, _T164, _T165, _T166] = +(X_T171[n, k0 + 2*x0, k1 + 2*x1, g + gci] * X_T51[k0, k1, g, gco, gci]), 500000000 + g < 1000000000, 500000000 + gci < 1000000000, 500000000 + gco < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T172    X_T171     X_T51  
//        g       128         1         1         1  
//       k0         3         0      7296       384  
//       k1         3         0       128       128  
//       x0        28      3584     14592         0  
//       x1        28       128       256         0  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { g, k0, k1, x0, x1 }
// Ranges: { 128, 3, 3, 28, 28 }
// Out stride: { 1, 0, 0, 3584, 128 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7296, 128, 14592, 256 }
// Input 2 offset: 0
// Input 2 stride: { 1, 384, 128, 0, 0 }
// Tile size: { 32, 1, 3, 4, 28 }
// Contraction output var shape: fp32(1, 28, 28, 128, 1):(100352, 3584, 128, 1, 1):392 KiB
// Computed true ops: 1806336
// Computed work groups: 28
// Computed inner loops: 3
// Computed shared mem: 29584
// Computed out regs: 14336
// Computed mem read: 29568
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 4, 1
__kernel void kernel_c25_sdk_41(__global float* restrict  X_T172, __global const float* restrict  in1, __global const float* restrict  in2)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[7300];
  __local float in2_shared[96];
  int g_gid = (get_group_id(1) * 32);
  int x0_gid = (get_group_id(0) * 4);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 1)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = (((g_gid + (k1_gid * 128)) + (k0_gid * 7296)) + (x0_gid * 14592));
        int g_tid = (tid % 32);
        int x0_tid = ((tid / 32) % 4);
        int k1_x1_k0_tid = ((tid / 128) % 2);
        for (int k1_x1_k0_lid = 0; k1_x1_k0_lid < 29; k1_x1_k0_lid += 1)
        {
          int k1_x1_k0_cond = ((k1_x1_k0_lid < 28) || (k1_x1_k0_tid < 1));
          if (k1_x1_k0_cond)
          {
            int k1_x1_k0 = ((2 * k1_x1_k0_lid) + k1_x1_k0_tid);
            int lidx = (((57 * g_tid) + (1825 * x0_tid)) + k1_x1_k0);
            int gidx = (((gbase + g_tid) + (14592 * x0_tid)) + (128 * k1_x1_k0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)415871)];
          }
        }
      }
      {
        int gbase = ((g_gid + (k1_gid * 128)) + (k0_gid * 384));
        int g_tid = (tid % 32);
        int k1_k0_tid = ((tid / 32) % 4);
        int k1_k0_cond = (k1_k0_tid < 3);
        if (k1_k0_cond)
        {
          if ((tid < 128))
          {
            int lidx = ((3 * g_tid) + k1_k0_tid);
            int gidx = ((gbase + g_tid) + (128 * k1_k0_tid));
            in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)1151)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
      {
        int g_tid = (tid % 32);
        int x1_tid = ((tid / 32) % 4);
        int x0_tid = ((tid / 128) % 2);
        for (int x1_lid = 0; x1_lid < 7; x1_lid += 1)
        {
          int x1 = ((4 * x1_lid) + x1_tid);
          for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
          {
            int x0 = ((2 * x0_lid) + x0_tid);
            float val1 = in1_shared[((((57 * g_tid) + k1_lid) + (2 * x1)) + (1825 * x0))];
            float val2 = in2_shared[((3 * g_tid) + k1_lid)];
            int agg_idx = (x1_lid + (x0_lid * 7));
            float agg_rhs = mad(val2, val1, agg[agg_idx]);
            agg[agg_idx] = agg_rhs;
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int g_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  for (int x1_lid = 0; x1_lid < 7; x1_lid += 1)
  {
    int x1 = ((4 * x1_lid) + x1_tid);
    for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
    {
      int x0 = ((2 * x0_lid) + x0_tid);
      float LX_T172 = agg[(x1_lid + (x0_lid * 7))];
      int gout_idx = (((g_gid + g_tid) + (3584 * (x0_gid + x0))) + (128 * x1));
      X_T172[gout_idx] = LX_T172;
    }
  }
}
