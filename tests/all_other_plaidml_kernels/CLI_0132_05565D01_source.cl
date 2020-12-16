#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 14 1
// lid: 256 1 1
// Original:
// X_T162[n0, n1, n2, a : _T179, _T180, _T181, _T182] = =(X_T161[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T162[n0, n1, n2, a : _T179, _T180, _T181, _T182] = =(X_T161[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= a < 192, 0 <= a < 224, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= a < 192 }
// Defracted:
// X_T162[n0, n1, n2, a : _T179, _T180, _T181, _T182] = =(X_T161[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T162    X_T161  
//        a       192         1         1  
//       n1        56     12544     10752  
//       n2        56       224       192  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 192, 56, 56 }
// Out stride: { 1, 12544, 224 }
// Input 1 offset: 0
// Input 1 stride: { 1, 10752, 192 }
// Tile size: { 192, 4, 4 }
// Contraction output var shape: fp32(1, 56, 56, 224):(702464, 12544, 224, 1):2744 KiB
// Computed true ops: 1204224
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 12304
// Computed out regs: 12288
// Computed mem read: 12288
// Computed mem write: 12288
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 14, 1
__kernel void kernel_c68_sdk_42(__global float* restrict  X_T162, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3076];
  int n2_gid = (get_group_id(0) * 4);
  int n1_gid = (get_group_id(1) * 4);
  {
    {
      int gbase = ((n2_gid * 192) + (n1_gid * 10752));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 3; a_n2_lid += 1)
      {
        int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
        for (int n1_lid = 0; n1_lid < 4; n1_lid += 1)
        {
          int lidx = (a_n2 + (769 * n1_lid));
          int gidx = ((gbase + a_n2) + (10752 * n1_lid));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)602111)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 4);
    int n1_tid = ((tid / 128) % 2);
    for (int a_lid = 0; a_lid < 6; a_lid += 1)
    {
      int a = ((32 * a_lid) + a_tid);
      for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
      {
        int n1 = ((2 * n1_lid) + n1_tid);
        float val1 = in1_shared[((a + (192 * n2_tid)) + (769 * n1))];
        int agg_idx = (a_lid + (n1_lid * 6));
        agg[agg_idx] = val1;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 4);
  int n1_tid = ((tid / 128) % 2);
  for (int a_lid = 0; a_lid < 6; a_lid += 1)
  {
    int a = ((32 * a_lid) + a_tid);
    for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
    {
      int n1 = ((2 * n1_lid) + n1_tid);
      float LX_T162 = agg[(a_lid + (n1_lid * 6))];
      int gout_idx = ((a + (12544 * (n1_gid + n1))) + (224 * (n2_gid + n2_tid)));
      X_T162[gout_idx] = LX_T162;
    }
  }
}
