#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 8960 1 1
// lid: 256 1 1
// Original:
// X_T240[n0, n1, n2, 64 + a : _T296, _T297, _T298, _T299] = =(X_T239[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T240[n0, n1, n2, 64 + a : _T296, _T297, _T298, _T299] = =(X_T239[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 35, 0 <= n2 < 35, 0 <= n1 < 35, 0 <= n2 < 35, 0 <= a < 64, 0 <= 64 + a < 128, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 35, 0 <= n2 < 35, 0 <= a < 64 }
// Defracted:
// X_T240[n0, n1, n2, 64 + a : _T296, _T297, _T298, _T299] = =(X_T239[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T240    X_T239  
//        a        64         1         1  
//       n1        35      4480      2240  
//       n2        35       128        64  
//      off                  64         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 64, 35, 35 }
// Out stride: { 1, 4480, 128 }
// Input 1 offset: 0
// Input 1 stride: { 1, 2240, 64 }
// Tile size: { 64, 1, 35 }
// Contraction output var shape: fp32(1, 35, 35, 128):(156800, 4480, 128, 1):612.5 KiB
// Computed true ops: 156800
// Computed work groups: 35
// Computed inner loops: 1
// Computed shared mem: 8960
// Computed out regs: 10240
// Computed mem read: 8960
// Computed mem write: 8960
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 8960, 1, 1
__kernel void kernel_c51_sdk_69(__global float* restrict  X_T240, __global const float* restrict  in1)
{
  X_T240 = (X_T240 + 64);
  int tid = get_local_id(0);
  float agg[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[2240];
  int n1_gid = get_group_id(0);
  {
    {
      int gbase = (n1_gid * 2240);
      int a_n2_n1_tid = (tid % 256);
      for (int a_n2_n1_lid = 0; a_n2_n1_lid < 9; a_n2_n1_lid += 1)
      {
        int a_n2_n1_cond = ((a_n2_n1_lid < 8) || (a_n2_n1_tid < 192));
        if (a_n2_n1_cond)
        {
          int a_n2_n1 = ((256 * a_n2_n1_lid) + a_n2_n1_tid);
          int gidx = (gbase + a_n2_n1);
          in1_shared[a_n2_n1] = in1[clamp((int)gidx, (int)0, (int)78399)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 8);
    for (int n2_lid = 0; n2_lid < 5; n2_lid += 1)
    {
      int n2_cond = ((n2_lid < 4) || (n2_tid < 3));
      int n2 = select((int)0, (int)((8 * n2_lid) + n2_tid), (int)n2_cond);
      for (int a_lid = 0; a_lid < 2; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float val1 = in1_shared[(a + (64 * n2))];
        int agg_idx = (a_lid + (n2_lid * 2));
        agg[agg_idx] = select((float)agg[agg_idx], (float)val1, (int)n2_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 8);
  for (int n2_lid = 0; n2_lid < 5; n2_lid += 1)
  {
    int n2_cond = ((n2_lid < 4) || (n2_tid < 3));
    if (n2_cond)
    {
      int n2 = ((8 * n2_lid) + n2_tid);
      for (int a_lid = 0; a_lid < 2; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float LX_T240 = agg[(a_lid + (n2_lid * 2))];
        int gout_idx = ((a + (4480 * n1_gid)) + (128 * n2));
        X_T240[gout_idx] = LX_T240;
      }
    }
  }
}
