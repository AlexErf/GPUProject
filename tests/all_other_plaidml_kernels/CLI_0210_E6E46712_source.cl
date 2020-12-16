#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Original:
// X_T249[n0, n1, n2, a : _T285, _T286, _T287, _T288] = =(X_T248[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T249[n0, n1, n2, a : _T285, _T286, _T287, _T288] = =(X_T248[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= a < 128, 0 <= a < 160, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= a < 128 }
// Defracted:
// X_T249[n0, n1, n2, a : _T285, _T286, _T287, _T288] = =(X_T248[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T249    X_T248  
//        a       128         1         1  
//       n1        28      4480      3584  
//       n2        28       160       128  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 128, 28, 28 }
// Out stride: { 1, 4480, 160 }
// Input 1 offset: 0
// Input 1 stride: { 1, 3584, 128 }
// Tile size: { 128, 28, 1 }
// Contraction output var shape: fp32(1, 28, 28, 160):(125440, 4480, 160, 1):490 KiB
// Computed true ops: 200704
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 14448
// Computed out regs: 16384
// Computed mem read: 14336
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c108_sdk_66(__global float* restrict  X_T249, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3612];
  int n2_gid = get_group_id(0);
  {
    {
      int gbase = (n2_gid * 128);
      int a_n2_tid = (tid % 128);
      int n1_tid = ((tid / 128) % 2);
      for (int n1_lid = 0; n1_lid < 14; n1_lid += 1)
      {
        int n1 = ((2 * n1_lid) + n1_tid);
        int lidx = (a_n2_tid + (129 * n1));
        int gidx = ((gbase + a_n2_tid) + (3584 * n1));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)100351)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n1_tid = ((tid / 32) % 8);
    for (int n1_lid = 0; n1_lid < 4; n1_lid += 1)
    {
      int n1_cond = ((n1_lid < 3) || (n1_tid < 4));
      int n1 = select((int)0, (int)((8 * n1_lid) + n1_tid), (int)n1_cond);
      for (int a_lid = 0; a_lid < 4; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float val1 = in1_shared[(a + (129 * n1))];
        int agg_idx = (a_lid + (n1_lid * 4));
        agg[agg_idx] = select((float)agg[agg_idx], (float)val1, (int)n1_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n1_tid = ((tid / 32) % 8);
  for (int n1_lid = 0; n1_lid < 4; n1_lid += 1)
  {
    int n1_cond = ((n1_lid < 3) || (n1_tid < 4));
    if (n1_cond)
    {
      int n1 = ((8 * n1_lid) + n1_tid);
      for (int a_lid = 0; a_lid < 4; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float LX_T249 = agg[(a_lid + (n1_lid * 4))];
        int gout_idx = ((a + (4480 * n1)) + (160 * n2_gid));
        X_T249[gout_idx] = LX_T249;
      }
    }
  }
}
