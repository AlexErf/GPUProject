#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 4 1
// lid: 256 1 1
// Original:
// X_T1165[n, x0, x1, c : _T1662, _T1663, _T1664, _T1665] = +(X_T1163[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2
// With Index Variables Made Integral:
// X_T1165[n, x0, x1, c : _T1662, _T1663, _T1664, _T1665] = +(X_T1163[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 2, 0 <= k1 < 2, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= k0 + 2*x0 < 14, 0 <= k1 + 2*x1 < 14, 0 <= c < 512, 0 <= c < 512, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 2, 0 <= k1 < 2, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= k0 + 2*x0 < 14, 0 <= k1 + 2*x1 < 14, 0 <= c < 512 }
// Defracted:
// X_T1165[n, x0, x1, c : _T1662, _T1663, _T1664, _T1665] = +(X_T1163[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 2, k1 < 2, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T1165   X_T1163  
//        c       512         1         1  
//       k0         2         0      7168  
//       k1         2         0       512  
//       x0         7      3584     14336  
//       x1         7       512      1024  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 512, 2, 2, 7, 7 }
// Out stride: { 1, 0, 0, 3584, 512 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7168, 512, 14336, 1024 }
// Tile size: { 128, 1, 1, 1, 7 }
// Contraction output var shape: fp32(1, 7, 7, 512):(25088, 3584, 512, 1):98 KiB
// Computed true ops: 200704
// Computed work groups: 28
// Computed inner loops: 4
// Computed shared mem: 3584
// Computed out regs: 4096
// Computed mem read: 3584
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 4, 1
__kernel void kernel_c68_sdk_399(__global float* restrict  X_T1165, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[896];
  int c_gid = (get_group_id(1) * 128);
  int x0_gid = get_group_id(0);
  for (int k0_gid = 0; k0_gid < 2; k0_gid += 1)
  {
    for (int k1_gid = 0; k1_gid < 2; k1_gid += 1)
    {
      {
        int gbase = (((c_gid + (k1_gid * 512)) + (k0_gid * 7168)) + (x0_gid * 14336));
        int c_tid = (tid % 128);
        int x1_k0_tid = ((tid / 128) % 2);
        for (int x1_k0_lid = 0; x1_k0_lid < 4; x1_k0_lid += 1)
        {
          int x1_k0_cond = ((x1_k0_lid < 3) || (x1_k0_tid < 1));
          if (x1_k0_cond)
          {
            int x1_k0 = ((2 * x1_k0_lid) + x1_k0_tid);
            int lidx = ((7 * c_tid) + x1_k0);
            int gidx = ((gbase + c_tid) + (1024 * x1_k0));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)100351)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      int c_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 8);
      int x1_cond = (x1_tid < 7);
      int x1 = select((int)0, (int)x1_tid, (int)x1_cond);
      for (int c_lid = 0; c_lid < 4; c_lid += 1)
      {
        int c = ((32 * c_lid) + c_tid);
        float val1 = in1_shared[((7 * c) + x1)];
        float agg_rhs = (agg[c_lid] + val1);
        agg[c_lid] = select((float)agg[c_lid], (float)agg_rhs, (int)x1_cond);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 8);
  int x1_cond = (x1_tid < 7);
  if (x1_cond)
  {
    for (int c_lid = 0; c_lid < 4; c_lid += 1)
    {
      int c = ((32 * c_lid) + c_tid);
      float LX_T1165 = agg[c_lid];
      int gout_idx = (((c_gid + c) + (3584 * x0_gid)) + (512 * x1_tid));
      X_T1165[gout_idx] = LX_T1165;
    }
  }
}
