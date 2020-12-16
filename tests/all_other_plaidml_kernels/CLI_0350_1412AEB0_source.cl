#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 2 1
// lid: 256 1 1
// Original:
// X_T781[n0, n1, n2, a : _T1056, _T1057, _T1058, _T1059] = =(X_T780[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T781[n0, n1, n2, a : _T1056, _T1057, _T1058, _T1059] = =(X_T780[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 512, 0 <= a < 544, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 512 }
// Defracted:
// X_T781[n0, n1, n2, a : _T1056, _T1057, _T1058, _T1059] = =(X_T780[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T781    X_T780  
//        a       512         1         1  
//       n1        14      7616      7168  
//       n2        14       544       512  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 512, 14, 14 }
// Out stride: { 1, 7616, 544 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7168, 512 }
// Tile size: { 256, 14, 1 }
// Contraction output var shape: fp32(1, 14, 14, 544):(106624, 7616, 544, 1):416.5 KiB
// Computed true ops: 200704
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 14392
// Computed out regs: 16384
// Computed mem read: 14336
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 2, 1
__kernel void kernel_c124_sdk_252(__global float* restrict  X_T781, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3598];
  int a_gid = (get_group_id(1) * 256);
  int n2_gid = get_group_id(0);
  {
    {
      int gbase = (a_gid + (n2_gid * 512));
      int a_tid = (tid % 256);
      for (int n1_lid = 0; n1_lid < 14; n1_lid += 1)
      {
        int lidx = (a_tid + (257 * n1_lid));
        int gidx = ((gbase + a_tid) + (7168 * n1_lid));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)100351)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n1_tid = ((tid / 32) % 8);
    for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
    {
      int n1_cond = ((n1_lid < 1) || (n1_tid < 6));
      int n1 = select((int)0, (int)((8 * n1_lid) + n1_tid), (int)n1_cond);
      for (int a_lid = 0; a_lid < 8; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float val1 = in1_shared[(a + (257 * n1))];
        int agg_idx = (a_lid + (n1_lid * 8));
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
      for (int a_lid = 0; a_lid < 8; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float LX_T781 = agg[(a_lid + (n1_lid * 8))];
        int gout_idx = (((a_gid + a) + (7616 * n1)) + (544 * n2_gid));
        X_T781[gout_idx] = LX_T781;
      }
    }
  }
}
