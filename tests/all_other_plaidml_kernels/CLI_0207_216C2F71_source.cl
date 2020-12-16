#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 21 1
// lid: 256 1 1
// Original:
// X_T1761[n, x0, x1, c : _T2770, _T2771, _T2772, _T2773] = +(X_T1385[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1
// With Index Variables Made Integral:
// X_T1761[n, x0, x1, c : _T2770, _T2771, _T2772, _T2773] = +(X_T1385[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= k0 < 1, 0 <= k1 < 1, 0 <= n < 1, 0 <= n < 1, 0 <= x0 < 21, 0 <= x1 < 21, 0 <= k0 + 2*x0 < 42, 0 <= k1 + 2*x1 < 42, 0 <= c < 1008, 0 <= c < 1008, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= k0 < 1, 0 <= k1 < 1, 0 <= n < 1, 0 <= x0 < 21, 0 <= x1 < 21, 0 <= k0 + 2*x0 < 42, 0 <= k1 + 2*x1 < 42, 0 <= c < 1008 }
// Defracted:
// X_T1761[n, x0, x1, c : _T2770, _T2771, _T2772, _T2773] = +(X_T1385[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T1761   X_T1385  
//        c      1008         1         1  
//       x0        21     21168     84672  
//       x1        21      1008      2016  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, x0, x1 }
// Ranges: { 1008, 21, 21 }
// Out stride: { 1, 21168, 1008 }
// Input 1 offset: 0
// Input 1 stride: { 1, 84672, 2016 }
// Tile size: { 1008, 1, 1 }
// Contraction output var shape: fp32(1, 21, 21, 1008):(444528, 21168, 1008, 1):1736.44 KiB
// Computed true ops: 889056
// Computed work groups: 441
// Computed inner loops: 1
// Computed shared mem: 4032
// Computed out regs: 4096
// Computed mem read: 3968
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 21, 1
__kernel void kernel_c42_sdk_670(__global float* restrict  X_T1761, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[1008];
  int x1_gid = get_group_id(0);
  int x0_gid = get_group_id(1);
  {
    {
      int gbase = ((x1_gid * 2016) + (x0_gid * 84672));
      int c_tid = (tid % 256);
      for (int c_lid = 0; c_lid < 4; c_lid += 1)
      {
        int c_cond = ((c_lid < 3) || (c_tid < 240));
        if (c_cond)
        {
          int c = ((256 * c_lid) + c_tid);
          int gidx = (gbase + c);
          in1_shared[c] = in1[clamp((int)gidx, (int)0, (int)1778111)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int c_tid = (tid % 256);
    for (int c_lid = 0; c_lid < 4; c_lid += 1)
    {
      int c_cond = ((c_lid < 3) || (c_tid < 240));
      int c = select((int)0, (int)((256 * c_lid) + c_tid), (int)c_cond);
      float val1 = in1_shared[c];
      float agg_rhs = (agg[c_lid] + val1);
      agg[c_lid] = select((float)agg[c_lid], (float)agg_rhs, (int)c_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 256);
  for (int c_lid = 0; c_lid < 4; c_lid += 1)
  {
    int c_cond = ((c_lid < 3) || (c_tid < 240));
    if (c_cond)
    {
      int c = ((256 * c_lid) + c_tid);
      float LX_T1761 = agg[c_lid];
      int gout_idx = ((c + (21168 * x0_gid)) + (1008 * x1_gid));
      X_T1761[gout_idx] = LX_T1761;
    }
  }
}
