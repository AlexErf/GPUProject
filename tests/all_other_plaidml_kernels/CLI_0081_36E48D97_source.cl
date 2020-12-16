#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 14592 1 1
// lid: 256 1 1
// Original:
// X_T98[n, o0, o1, c : _T104, _T105, _T106, _T107] = =(X_T97[])
// With Index Variables Made Integral:
// X_T98[n, o0, o1, c : _T104, _T105, _T106, _T107] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= c < 11, 0 <= o0 < 113, 0 <= o1 < 113, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= c < 11, 0 <= o0 < 113, 0 <= o1 < 113 }
// Defracted:
// X_T98[n, o0, o1, c : _T104, _T105, _T106, _T107] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range     X_T98     X_T97  
//        c        11         1         0  
//       o0       113      1243         0  
//       o1       113        11         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 11, 113, 113 }
// Out stride: { 1, 1243, 11 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 11, 113, 2 }
// Contraction output var shape: fp32(1, 113, 113, 11):(140459, 1243, 11, 1):548.668 KiB
// Computed true ops: 280918
// Computed work groups: 57
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 15360
// Computed mem read: 128
// Computed mem write: 28928
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 14592, 1, 1
__kernel void kernel_c42_sdk_21(__global float* restrict  X_T98)
{
  int tid = get_local_id(0);
  float agg[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  int o1_gid = (get_group_id(0) * 2);
  {
    int c_tid = (tid % 16);
    int o1_tid = ((tid / 16) % 2);
    int o0_tid = ((tid / 32) % 8);
    int c_cond = (c_tid < 11);
    int c = select((int)0, (int)c_tid, (int)c_cond);
    for (int o0_lid = 0; o0_lid < 15; o0_lid += 1)
    {
      int o0_cond = ((o0_lid < 14) || (o0_tid < 1));
      int o0 = select((int)0, (int)((8 * o0_lid) + o0_tid), (int)(c_cond && o0_cond));
      float val1 = 1.0f;
      agg[o0_lid] = select((float)agg[o0_lid], (float)val1, (int)(c_cond && o0_cond));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 16);
  int o1_tid = ((tid / 16) % 2);
  int o0_tid = ((tid / 32) % 8);
  int o1_cond = ((o1_gid != 112) || (o1_tid < 1));
  if (o1_cond)
  {
    int c_cond = (c_tid < 11);
    if (c_cond)
    {
      for (int o0_lid = 0; o0_lid < 15; o0_lid += 1)
      {
        int o0_cond = ((o0_lid < 14) || (o0_tid < 1));
        if (o0_cond)
        {
          int o0 = ((8 * o0_lid) + o0_tid);
          float LX_T98 = agg[o0_lid];
          int gout_idx = ((c_tid + (1243 * o0)) + (11 * (o1_gid + o1_tid)));
          X_T98[gout_idx] = LX_T98;
        }
      }
    }
  }
}
