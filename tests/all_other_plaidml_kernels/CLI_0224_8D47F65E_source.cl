#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 2 1
// lid: 256 1 1
// Original:
// X_T549[n0, n1, n2, a : _T760, _T761, _T762, _T763] = =(X_T548[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T549[n0, n1, n2, a : _T760, _T761, _T762, _T763] = =(X_T548[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 256, 0 <= a < 288, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 256 }
// Defracted:
// X_T549[n0, n1, n2, a : _T760, _T761, _T762, _T763] = =(X_T548[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T549    X_T548  
//        a       256         1         1  
//       n1        14      4032      3584  
//       n2        14       288       256  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 256, 14, 14 }
// Out stride: { 1, 4032, 288 }
// Input 1 offset: 0
// Input 1 stride: { 1, 3584, 256 }
// Tile size: { 128, 14, 1 }
// Contraction output var shape: fp32(1, 14, 14, 288):(56448, 4032, 288, 1):220.5 KiB
// Computed true ops: 100352
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 7224
// Computed out regs: 8192
// Computed mem read: 7168
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 2, 1
__kernel void kernel_c68_sdk_180(__global float* restrict  X_T549, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[8] = {0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1806];
  int a_gid = (get_group_id(1) * 128);
  int n2_gid = get_group_id(0);
  {
    {
      int gbase = (a_gid + (n2_gid * 256));
      int a_tid = (tid % 128);
      int n1_tid = ((tid / 128) % 2);
      for (int n1_lid = 0; n1_lid < 7; n1_lid += 1)
      {
        int n1 = ((2 * n1_lid) + n1_tid);
        int lidx = (a_tid + (129 * n1));
        int gidx = ((gbase + a_tid) + (3584 * n1));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)50175)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n1_tid = ((tid / 32) % 8);
    for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
    {
      int n1_cond = ((n1_lid < 1) || (n1_tid < 6));
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
  for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
  {
    int n1_cond = ((n1_lid < 1) || (n1_tid < 6));
    if (n1_cond)
    {
      int n1 = ((8 * n1_lid) + n1_tid);
      for (int a_lid = 0; a_lid < 4; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float LX_T549 = agg[(a_lid + (n1_lid * 4))];
        int gout_idx = (((a_gid + a) + (4032 * n1)) + (288 * n2_gid));
        X_T549[gout_idx] = LX_T549;
      }
    }
  }
}
