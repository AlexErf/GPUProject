#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Original:
// X_T257[n, x0, x1, c : _T350, _T351, _T352, _T353] = +(X_T45[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1
// With Index Variables Made Integral:
// X_T257[n, x0, x1, c : _T350, _T351, _T352, _T353] = +(X_T45[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= k0 < 1, 0 <= k1 < 1, 0 <= n < 1, 0 <= n < 1, 0 <= c < 32, 0 <= c < 32, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + 2*x0 < 111, 0 <= k1 + 2*x1 < 111, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= k0 < 1, 0 <= k1 < 1, 0 <= n < 1, 0 <= c < 32, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + 2*x0 < 111, 0 <= k1 + 2*x1 < 111 }
// Defracted:
// X_T257[n, x0, x1, c : _T350, _T351, _T352, _T353] = +(X_T45[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T257     X_T45  
//        c        32         1         1  
//       x0        56      1792      7104  
//       x1        56        32        64  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, x0, x1 }
// Ranges: { 32, 56, 56 }
// Out stride: { 1, 1792, 32 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7104, 64 }
// Tile size: { 32, 56, 2 }
// Contraction output var shape: fp32(1, 56, 56, 32):(100352, 1792, 32, 1):392 KiB
// Computed true ops: 200704
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 14600
// Computed out regs: 14336
// Computed mem read: 14336
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c42_sdk_81(__global float* restrict  X_T257, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3650];
  int x1_gid = (get_group_id(0) * 2);
  {
    {
      int gbase = (x1_gid * 64);
      int c_tid = (tid % 32);
      int x1_tid = ((tid / 32) % 2);
      int x0_tid = ((tid / 64) % 4);
      for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
      {
        int x0 = ((4 * x0_lid) + x0_tid);
        int lidx = (((57 * c_tid) + (1825 * x1_tid)) + x0);
        int gidx = (((gbase + c_tid) + (64 * x1_tid)) + (7104 * x0));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)394271)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int c_tid = (tid % 32);
    int x1_tid = ((tid / 32) % 2);
    int x0_tid = ((tid / 64) % 4);
    for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
    {
      int x0 = ((4 * x0_lid) + x0_tid);
      float val1 = in1_shared[(((57 * c_tid) + (1825 * x1_tid)) + x0)];
      float agg_rhs = (agg[x0_lid] + val1);
      agg[x0_lid] = agg_rhs;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 2);
  int x0_tid = ((tid / 64) % 4);
  for (int x0_lid = 0; x0_lid < 14; x0_lid += 1)
  {
    int x0 = ((4 * x0_lid) + x0_tid);
    float LX_T257 = agg[x0_lid];
    int gout_idx = ((c_tid + (1792 * x0)) + (32 * (x1_gid + x1_tid)));
    X_T257[gout_idx] = LX_T257;
  }
}
