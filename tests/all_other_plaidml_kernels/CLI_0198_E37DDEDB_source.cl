#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 11 11
// lid: 256 1 1
// Original:
// X_T1706[n, o0, o1, c : _T2685, _T2686, _T2687, _T2688] = =(X_T100[])
// With Index Variables Made Integral:
// X_T1706[n, o0, o1, c : _T2685, _T2686, _T2687, _T2688] = =(X_T100[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 21, 0 <= o1 < 21, 0 <= c < 336, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 21, 0 <= o1 < 21, 0 <= c < 336 }
// Defracted:
// X_T1706[n, o0, o1, c : _T2685, _T2686, _T2687, _T2688] = =(X_T100[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range   X_T1706    X_T100  
//        c       336         1         0  
//       o0        21      7056         0  
//       o1        21       336         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 336, 21, 21 }
// Out stride: { 1, 7056, 336 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 32, 2, 8 }
// Contraction output var shape: fp32(1, 21, 21, 336):(148176, 7056, 336, 1):578.812 KiB
// Computed true ops: 296352
// Computed work groups: 363
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 2048
// Computed mem read: 128
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 11, 11
__kernel void kernel_c42_sdk_648(__global float* restrict  X_T1706)
{
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  int c_gid = (get_group_id(1) * 32);
  int o1_gid = (get_group_id(0) * 8);
  int o0_gid = (get_group_id(2) * 2);
  {
    int c_tid = (tid % 32);
    int o1_tid = ((tid / 32) % 4);
    int o0_tid = ((tid / 128) % 2);
    for (int o1_lid = 0; o1_lid < 2; o1_lid += 1)
    {
      int o1 = ((4 * o1_lid) + o1_tid);
      float val1 = 1.0f;
      agg[o1_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o1_tid = ((tid / 32) % 4);
  int o0_tid = ((tid / 128) % 2);
  int c_cond = ((c_gid != 320) || (c_tid < 16));
  if (c_cond)
  {
    for (int o1_lid = 0; o1_lid < 2; o1_lid += 1)
    {
      int o1_cond = ((o1_lid < 1) || ((o1_gid != 16) || (o1_tid < 1)));
      if (o1_cond)
      {
        int o1 = ((4 * o1_lid) + o1_tid);
        int o0_cond = ((o0_gid != 20) || (o0_tid < 1));
        if (o0_cond)
        {
          float LX_T1706 = agg[o1_lid];
          int gout_idx = (((c_gid + c_tid) + (7056 * (o0_gid + o0_tid))) + (336 * (o1_gid + o1)));
          X_T1706[gout_idx] = LX_T1706;
        }
      }
    }
  }
}
