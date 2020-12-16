#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 7 1
// lid: 256 1 1
// Original:
// X_T1359[n, o0, o1, c : _T2129, _T2130, _T2131, _T2132] = =(X_T97[])
// With Index Variables Made Integral:
// X_T1359[n, o0, o1, c : _T2129, _T2130, _T2131, _T2132] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 14, 0 <= o1 < 14, 0 <= c < 88, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 14, 0 <= o1 < 14, 0 <= c < 88 }
// Defracted:
// X_T1359[n, o0, o1, c : _T2129, _T2130, _T2131, _T2132] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range   X_T1359     X_T97  
//        c        88         1         0  
//       o0        14      1232         0  
//       o1        14        88         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 88, 14, 14 }
// Out stride: { 1, 1232, 88 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 32, 2, 14 }
// Contraction output var shape: fp32(1, 14, 14, 88):(17248, 1232, 88, 1):67.375 KiB
// Computed true ops: 34496
// Computed work groups: 21
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 4096
// Computed mem read: 128
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 7, 1
__kernel void kernel_c42_sdk_510(__global float* restrict  X_T1359)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  int c_gid = (get_group_id(0) * 32);
  int o0_gid = (get_group_id(1) * 2);
  {
    int c_tid = (tid % 32);
    int o1_tid = ((tid / 32) % 4);
    int o0_tid = ((tid / 128) % 2);
    for (int o1_lid = 0; o1_lid < 4; o1_lid += 1)
    {
      int o1_cond = ((o1_lid < 3) || (o1_tid < 2));
      int o1 = select((int)0, (int)((4 * o1_lid) + o1_tid), (int)o1_cond);
      float val1 = 1.0f;
      agg[o1_lid] = select((float)agg[o1_lid], (float)val1, (int)o1_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o1_tid = ((tid / 32) % 4);
  int o0_tid = ((tid / 128) % 2);
  int c_cond = ((c_gid != 64) || (c_tid < 24));
  if (c_cond)
  {
    for (int o1_lid = 0; o1_lid < 4; o1_lid += 1)
    {
      int o1_cond = ((o1_lid < 3) || (o1_tid < 2));
      if (o1_cond)
      {
        int o1 = ((4 * o1_lid) + o1_tid);
        float LX_T1359 = agg[o1_lid];
        int gout_idx = (((c_gid + c_tid) + (1232 * (o0_gid + o0_tid))) + (88 * o1));
        X_T1359[gout_idx] = LX_T1359;
      }
    }
  }
}
