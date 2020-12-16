#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T308[n0, n1, n2, a : _T396, _T397, _T398, _T399] = =(X_T307[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T308[n0, n1, n2, a : _T396, _T397, _T398, _T399] = =(X_T307[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= a < 224, 0 <= a < 256, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= a < 224 }
// Defracted:
// X_T308[n0, n1, n2, a : _T396, _T397, _T398, _T399] = =(X_T307[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T308    X_T307  
//        a       224         1         1  
//       n1        28      7168      6272  
//       n2        28       256       224  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 224, 28, 28 }
// Out stride: { 1, 7168, 256 }
// Input 1 offset: 0
// Input 1 stride: { 1, 6272, 224 }
// Tile size: { 32, 4, 28 }
// Contraction output var shape: fp32(1, 28, 28, 256):(200704, 7168, 256, 1):784 KiB
// Computed true ops: 351232
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 14464
// Computed out regs: 14336
// Computed mem read: 14336
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c68_sdk_93(__global float* restrict  X_T308, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3616];
  int a_gid = (get_group_id(0) * 32);
  int n1_gid = (get_group_id(1) * 4);
  {
    {
      int gbase = (a_gid + (n1_gid * 6272));
      int a_tid = (tid % 32);
      int n2_n1_tid = ((tid / 32) % 8);
      for (int n2_n1_lid = 0; n2_n1_lid < 14; n2_n1_lid += 1)
      {
        int n2_n1 = ((8 * n2_n1_lid) + n2_n1_tid);
        int lidx = ((113 * a_tid) + n2_n1);
        int gidx = ((gbase + a_tid) + (224 * n2_n1));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)175615)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 4);
    int n1_tid = ((tid / 128) % 2);
    for (int n2_lid = 0; n2_lid < 7; n2_lid += 1)
    {
      int n2 = ((4 * n2_lid) + n2_tid);
      for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
      {
        int n1 = ((2 * n1_lid) + n1_tid);
        float val1 = in1_shared[(((113 * a_tid) + n2) + (28 * n1))];
        int agg_idx = (n2_lid + (n1_lid * 7));
        agg[agg_idx] = val1;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 4);
  int n1_tid = ((tid / 128) % 2);
  for (int n2_lid = 0; n2_lid < 7; n2_lid += 1)
  {
    int n2 = ((4 * n2_lid) + n2_tid);
    for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
    {
      int n1 = ((2 * n1_lid) + n1_tid);
      float LX_T308 = agg[(n2_lid + (n1_lid * 7))];
      int gout_idx = (((a_gid + a_tid) + (7168 * (n1_gid + n1))) + (256 * n2));
      X_T308[gout_idx] = LX_T308;
    }
  }
}
