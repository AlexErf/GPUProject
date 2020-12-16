#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 83 1
// lid: 256 1 1
// Original:
// X_T3042[i0, i1, i2, i3 : _T4848, _T4849, _T4850, _T4851] = =(X_T3041[i0, 1 + i1, 1 + i2, i3])
// With Index Variables Made Integral:
// X_T3042[i0, i1, i2, i3 : _T4848, _T4849, _T4850, _T4851] = =(X_T3041[i0, 1 + i1, 1 + i2, i3]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000, 500000000 + i2 < 1000000000, 500000000 + i3 < 1000000000
// Constraints:{ 0 <= i0 < 1, 0 <= i0 < 1, 0 <= i1 < 21, 0 <= i2 < 21, 0 <= 1 + i1 < 22, 0 <= 1 + i2 < 22, 0 <= i3 < 2016, 0 <= i3 < 2016, 0 <= 500000000 + i0 < 1000000000, 0 <= 500000000 + i1 < 1000000000, 0 <= 500000000 + i2 < 1000000000, 0 <= 500000000 + i3 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= i0 < 1, 0 <= i1 < 21, 0 <= i2 < 21, 0 <= i3 < 2016 }
// Defracted:
// X_T3042[i0, i1, i2, i3 : _T4848, _T4849, _T4850, _T4851] = =(X_T3041[i0, 1 + i1, 1 + i2, i3]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000, 500000000 + i2 < 1000000000, 500000000 + i3 < 1000000000
// Flattened:
//              Range   X_T3042   X_T3041  
//       i1        21     42336     44352  
//       i2        21      2016      2016  
//       i3      2016         1         1  
//      off                   0     46368  
//      vec                   1         1  
// 
// Names: { i1, i2_i3 }
// Ranges: { 21, 42336 }
// Out stride: { 42336, 1 }
// Input 1 offset: 46368
// Input 1 stride: { 44352, 1 }
// Tile size: { 1, 512 }
// Contraction output var shape: fp32(1, 21, 21, 2016):(889056, 42336, 2016, 1):3472.88 KiB
// Computed true ops: 1778112
// Computed work groups: 1743
// Computed inner loops: 1
// Computed shared mem: 2048
// Computed out regs: 2048
// Computed mem read: 2048
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 83, 1
__kernel void kernel_c42_sdk_1178(__global float* restrict  X_T3042, __global const float* restrict  in1)
{
  in1 = (in1 + 46368);
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[512];
  int i2_i3_gid = (get_group_id(1) * 512);
  int i1_gid = get_group_id(0);
  {
    {
      int gbase = (i2_i3_gid + (i1_gid * 44352));
      int i2_i3_tid = (tid % 256);
      for (int i2_i3_lid = 0; i2_i3_lid < 2; i2_i3_lid += 1)
      {
        int i2_i3 = ((256 * i2_i3_lid) + i2_i3_tid);
        int gidx = (gbase + i2_i3);
        in1_shared[i2_i3] = in1[clamp((int)gidx, (int)-46368, (int)929375)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int i2_i3_tid = (tid % 256);
    for (int i2_i3_lid = 0; i2_i3_lid < 2; i2_i3_lid += 1)
    {
      int i2_i3 = ((256 * i2_i3_lid) + i2_i3_tid);
      float val1 = in1_shared[i2_i3];
      agg[i2_i3_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int i2_i3_tid = (tid % 256);
  for (int i2_i3_lid = 0; i2_i3_lid < 2; i2_i3_lid += 1)
  {
    int i2_i3_cond = ((i2_i3_lid < 1) || ((i2_i3_gid != 41984) || (i2_i3_tid < 96)));
    if (i2_i3_cond)
    {
      int i2_i3 = ((256 * i2_i3_lid) + i2_i3_tid);
      float LX_T3042 = agg[i2_i3_lid];
      int gout_idx = ((42336 * (int)i1_gid) + (int)(i2_i3_gid + i2_i3));
      X_T3042[gout_idx] = LX_T3042;
    }
  }
}
