#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Original:
// X_T647[n, o0, o1, c : _T979, _T980, _T981, _T982] = =(X_T97[])
// With Index Variables Made Integral:
// X_T647[n, o0, o1, c : _T979, _T980, _T981, _T982] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 28, 0 <= o1 < 28, 0 <= c < 44, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 28, 0 <= o1 < 28, 0 <= c < 44 }
// Defracted:
// X_T647[n, o0, o1, c : _T979, _T980, _T981, _T982] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range    X_T647     X_T97  
//        c        44         1         0  
//       o0        28      1232         0  
//       o1        28        44         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 44, 28, 28 }
// Out stride: { 1, 1232, 44 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 44, 28, 1 }
// Contraction output var shape: fp32(1, 28, 28, 44):(34496, 1232, 44, 1):134.75 KiB
// Computed true ops: 68992
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 8192
// Computed mem read: 128
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c42_sdk_230(__global float* restrict  X_T647)
{
  int tid = get_local_id(0);
  float agg[8] = {0, 0, 0, 0, 0, 0, 0, 0, };
  int o1_gid = get_group_id(0);
  {
    int c_tid = (tid % 32);
    int o0_tid = ((tid / 32) % 8);
    for (int c_lid = 0; c_lid < 2; c_lid += 1)
    {
      int c_cond = ((c_lid < 1) || (c_tid < 12));
      int c = select((int)0, (int)((32 * c_lid) + c_tid), (int)c_cond);
      for (int o0_lid = 0; o0_lid < 4; o0_lid += 1)
      {
        int o0_cond = ((o0_lid < 3) || (o0_tid < 4));
        int o0 = select((int)0, (int)((8 * o0_lid) + o0_tid), (int)(c_cond && o0_cond));
        float val1 = 1.0f;
        int agg_idx = (c_lid + (o0_lid * 2));
        agg[agg_idx] = select((float)agg[agg_idx], (float)val1, (int)(c_cond && o0_cond));
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o0_tid = ((tid / 32) % 8);
  for (int c_lid = 0; c_lid < 2; c_lid += 1)
  {
    int c_cond = ((c_lid < 1) || (c_tid < 12));
    if (c_cond)
    {
      int c = ((32 * c_lid) + c_tid);
      for (int o0_lid = 0; o0_lid < 4; o0_lid += 1)
      {
        int o0_cond = ((o0_lid < 3) || (o0_tid < 4));
        if (o0_cond)
        {
          int o0 = ((8 * o0_lid) + o0_tid);
          float LX_T647 = agg[(c_lid + (o0_lid * 2))];
          int gout_idx = ((c + (1232 * o0)) + (44 * o1_gid));
          X_T647[gout_idx] = LX_T647;
        }
      }
    }
  }
}
