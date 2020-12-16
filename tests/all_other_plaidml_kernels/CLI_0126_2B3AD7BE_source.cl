#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 14 1
// lid: 256 1 1
// Original:
// X_T137[n0, n1, n2, a : _T142, _T143, _T144, _T145] = =(X_T136[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T137[n0, n1, n2, a : _T142, _T143, _T144, _T145] = =(X_T136[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= a < 160, 0 <= a < 192, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 56, 0 <= n2 < 56, 0 <= a < 160 }
// Defracted:
// X_T137[n0, n1, n2, a : _T142, _T143, _T144, _T145] = =(X_T136[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T137    X_T136  
//        a       160         1         1  
//       n1        56     10752      8960  
//       n2        56       192       160  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 160, 56, 56 }
// Out stride: { 1, 10752, 192 }
// Input 1 offset: 0
// Input 1 stride: { 1, 8960, 160 }
// Tile size: { 160, 4, 4 }
// Contraction output var shape: fp32(1, 56, 56, 192):(602112, 10752, 192, 1):2352 KiB
// Computed true ops: 1003520
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 10256
// Computed out regs: 10240
// Computed mem read: 10240
// Computed mem write: 10240
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 14, 1
__kernel void kernel_c68_sdk_33(__global float* restrict  X_T137, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[2564];
  int n2_gid = (get_group_id(0) * 4);
  int n1_gid = (get_group_id(1) * 4);
  {
    {
      int gbase = ((n2_gid * 160) + (n1_gid * 8960));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 3; a_n2_lid += 1)
      {
        int a_n2_cond = ((a_n2_lid < 2) || (a_n2_tid < 128));
        if (a_n2_cond)
        {
          int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
          for (int n1_lid = 0; n1_lid < 4; n1_lid += 1)
          {
            int lidx = (a_n2 + (641 * n1_lid));
            int gidx = ((gbase + a_n2) + (8960 * n1_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)501759)];
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 4);
    int n1_tid = ((tid / 128) % 2);
    for (int a_lid = 0; a_lid < 5; a_lid += 1)
    {
      int a = ((32 * a_lid) + a_tid);
      for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
      {
        int n1 = ((2 * n1_lid) + n1_tid);
        float val1 = in1_shared[((a + (160 * n2_tid)) + (641 * n1))];
        int agg_idx = (a_lid + (n1_lid * 5));
        agg[agg_idx] = val1;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 4);
  int n1_tid = ((tid / 128) % 2);
  for (int a_lid = 0; a_lid < 5; a_lid += 1)
  {
    int a = ((32 * a_lid) + a_tid);
    for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
    {
      int n1 = ((2 * n1_lid) + n1_tid);
      float LX_T137 = agg[(a_lid + (n1_lid * 5))];
      int gout_idx = ((a + (10752 * (n1_gid + n1))) + (192 * (n2_gid + n2_tid)));
      X_T137[gout_idx] = LX_T137;
    }
  }
}
