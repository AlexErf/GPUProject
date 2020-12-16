#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 2 1
// lid: 256 1 1
// Original:
// X_T565[n, x0, x1, c : _T743, _T744, _T745, _T746] = +(X_T563[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2
// With Index Variables Made Integral:
// X_T565[n, x0, x1, c : _T743, _T744, _T745, _T746] = +(X_T563[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 2, 0 <= k1 < 2, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + 2*x0 < 28, 0 <= k1 + 2*x1 < 28, 0 <= c < 256, 0 <= c < 256, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 2, 0 <= k1 < 2, 0 <= x0 < 14, 0 <= x1 < 14, 0 <= k0 + 2*x0 < 28, 0 <= k1 + 2*x1 < 28, 0 <= c < 256 }
// Defracted:
// X_T565[n, x0, x1, c : _T743, _T744, _T745, _T746] = +(X_T563[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T565    X_T563  
//        c       256         1         1  
//       k0         2         0      7168  
//       k1         2         0       256  
//       x0        14      3584     14336  
//       x1        14       256       512  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 256, 2, 2, 14, 14 }
// Out stride: { 1, 0, 0, 3584, 256 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7168, 256, 14336, 512 }
// Tile size: { 128, 2, 1, 14, 1 }
// Contraction output var shape: fp32(1, 14, 14, 256):(50176, 3584, 256, 1):196 KiB
// Computed true ops: 401408
// Computed work groups: 28
// Computed inner loops: 2
// Computed shared mem: 14448
// Computed out regs: 8192
// Computed mem read: 14336
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 2, 1
__kernel void kernel_c108_sdk_177(__global float* restrict  X_T565, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[8] = {0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3612];
  int c_gid = (get_group_id(1) * 128);
  int x1_gid = get_group_id(0);
  for (int k0_gid = 0; k0_gid < 2; k0_gid += 2)
  {
    for (int k1_gid = 0; k1_gid < 2; k1_gid += 1)
    {
      {
        int gbase = (((c_gid + (k1_gid * 256)) + (x1_gid * 512)) + (k0_gid * 7168));
        int c_tid = (tid % 128);
        int k0_x0_tid = ((tid / 128) % 2);
        for (int k0_x0_lid = 0; k0_x0_lid < 14; k0_x0_lid += 1)
        {
          int k0_x0 = ((2 * k0_x0_lid) + k0_x0_tid);
          int lidx = (c_tid + (129 * k0_x0));
          int gidx = ((gbase + c_tid) + (7168 * k0_x0));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)200703)];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k0_lid = 0; k0_lid < 2; k0_lid += 1)
      {
        int c_tid = (tid % 32);
        int x0_tid = ((tid / 32) % 8);
        for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
        {
          int x0_cond = ((x0_lid < 1) || (x0_tid < 6));
          int x0 = select((int)0, (int)((8 * x0_lid) + x0_tid), (int)x0_cond);
          for (int c_lid = 0; c_lid < 4; c_lid += 1)
          {
            int c = ((32 * c_lid) + c_tid);
            float val1 = in1_shared[((c + (129 * k0_lid)) + (258 * x0))];
            int agg_idx = (c_lid + (x0_lid * 4));
            float agg_rhs = (agg[agg_idx] + val1);
            agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)x0_cond);
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 32);
  int x0_tid = ((tid / 32) % 8);
  for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
  {
    int x0_cond = ((x0_lid < 1) || (x0_tid < 6));
    if (x0_cond)
    {
      int x0 = ((8 * x0_lid) + x0_tid);
      for (int c_lid = 0; c_lid < 4; c_lid += 1)
      {
        int c = ((32 * c_lid) + c_tid);
        float LX_T565 = agg[(c_lid + (x0_lid * 4))];
        int gout_idx = (((c_gid + c) + (3584 * x0)) + (256 * x1_gid));
        X_T565[gout_idx] = LX_T565;
      }
    }
  }
}
