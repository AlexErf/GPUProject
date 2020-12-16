#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 17 1
// lid: 256 1 1
// Original:
// X_T2343[n, o0, o1, c : _T3712, _T3713, _T3714, _T3715] = =(X_T97[])
// With Index Variables Made Integral:
// X_T2343[n, o0, o1, c : _T3712, _T3713, _T3714, _T3715] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 14, 0 <= o1 < 14, 0 <= c < 528, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 14, 0 <= o1 < 14, 0 <= c < 528 }
// Defracted:
// X_T2343[n, o0, o1, c : _T3712, _T3713, _T3714, _T3715] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range   X_T2343     X_T97  
//        c       528         1         0  
//       o0        14      7392         0  
//       o1        14       528         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 528, 14, 14 }
// Out stride: { 1, 7392, 528 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 32, 2, 14 }
// Contraction output var shape: fp32(1, 14, 14, 528):(103488, 7392, 528, 1):404.25 KiB
// Computed true ops: 206976
// Computed work groups: 119
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 4096
// Computed mem read: 128
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 17, 1
__kernel void kernel_c42_sdk_897(__global float* restrict  X_T2343)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  int c_gid = (get_group_id(1) * 32);
  int o0_gid = (get_group_id(0) * 2);
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
  int c_cond = ((c_gid != 512) || (c_tid < 16));
  if (c_cond)
  {
    for (int o1_lid = 0; o1_lid < 4; o1_lid += 1)
    {
      int o1_cond = ((o1_lid < 3) || (o1_tid < 2));
      if (o1_cond)
      {
        int o1 = ((4 * o1_lid) + o1_tid);
        float LX_T2343 = agg[o1_lid];
        int gout_idx = (((c_gid + c_tid) + (7392 * (o0_gid + o0_tid))) + (528 * o1));
        X_T2343[gout_idx] = LX_T2343;
      }
    }
  }
}
