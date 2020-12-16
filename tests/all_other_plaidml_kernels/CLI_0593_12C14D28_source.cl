#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T1794[n, o0, o1, c : _T2556, _T2557, _T2558, _T2559] = =(X_T253[])
// With Index Variables Made Integral:
// X_T1794[n, o0, o1, c : _T2556, _T2557, _T2558, _T2559] = =(X_T253[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 14, 0 <= o1 < 14, 0 <= c < 896, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 14, 0 <= o1 < 14, 0 <= c < 896 }
// Defracted:
// X_T1794[n, o0, o1, c : _T2556, _T2557, _T2558, _T2559] = =(X_T253[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range   X_T1794    X_T253  
//        c       896         1         0  
//       o0        14     12544         0  
//       o1        14       896         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 896, 14, 14 }
// Out stride: { 1, 12544, 896 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 128, 2, 14 }
// Contraction output var shape: fp32(1, 14, 14, 896):(175616, 12544, 896, 1):686 KiB
// Computed true ops: 351232
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 16384
// Computed mem read: 128
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c124_sdk_616(__global float* restrict  X_T1794)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  int c_gid = (get_group_id(0) * 128);
  int o0_gid = (get_group_id(1) * 2);
  {
    int c_tid = (tid % 32);
    int o1_tid = ((tid / 32) % 4);
    int o0_tid = ((tid / 128) % 2);
    for (int o1_lid = 0; o1_lid < 4; o1_lid += 1)
    {
      int o1_cond = ((o1_lid < 3) || (o1_tid < 2));
      int o1 = select((int)0, (int)((4 * o1_lid) + o1_tid), (int)o1_cond);
      for (int c_lid = 0; c_lid < 4; c_lid += 1)
      {
        int c = ((32 * c_lid) + c_tid);
        float val1 = 1.0f;
        int agg_idx = (c_lid + (o1_lid * 4));
        agg[agg_idx] = select((float)agg[agg_idx], (float)val1, (int)o1_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o1_tid = ((tid / 32) % 4);
  int o0_tid = ((tid / 128) % 2);
  for (int o1_lid = 0; o1_lid < 4; o1_lid += 1)
  {
    int o1_cond = ((o1_lid < 3) || (o1_tid < 2));
    if (o1_cond)
    {
      int o1 = ((4 * o1_lid) + o1_tid);
      for (int c_lid = 0; c_lid < 4; c_lid += 1)
      {
        int c = ((32 * c_lid) + c_tid);
        float LX_T1794 = agg[(c_lid + (o1_lid * 4))];
        int gout_idx = (((c_gid + c) + (12544 * (o0_gid + o0_tid))) + (896 * o1));
        X_T1794[gout_idx] = LX_T1794;
      }
    }
  }
}
