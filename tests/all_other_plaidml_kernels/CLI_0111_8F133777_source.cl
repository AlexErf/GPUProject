#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7936 165 1
// lid: 256 1 1
// Original:
// X_T272[i0, i1, i2, i3 : _T380, _T381, _T382, _T383] = =(X_T271[i0, 1 + i1, 1 + i2, i3])
// With Index Variables Made Integral:
// X_T272[i0, i1, i2, i3 : _T380, _T381, _T382, _T383] = =(X_T271[i0, 1 + i1, 1 + i2, i3]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000, 500000000 + i2 < 1000000000, 500000000 + i3 < 1000000000
// Constraints:{ 0 <= i0 < 1, 0 <= i0 < 1, 0 <= i3 < 96, 0 <= i3 < 96, 0 <= i1 < 165, 0 <= i2 < 165, 0 <= 1 + i1 < 166, 0 <= 1 + i2 < 166, 0 <= 500000000 + i0 < 1000000000, 0 <= 500000000 + i1 < 1000000000, 0 <= 500000000 + i2 < 1000000000, 0 <= 500000000 + i3 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= i0 < 1, 0 <= i3 < 96, 0 <= i1 < 165, 0 <= i2 < 165 }
// Defracted:
// X_T272[i0, i1, i2, i3 : _T380, _T381, _T382, _T383] = =(X_T271[i0, 1 + i1, 1 + i2, i3]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000, 500000000 + i2 < 1000000000, 500000000 + i3 < 1000000000
// Flattened:
//              Range    X_T272    X_T271  
//       i1       165     15840     15936  
//       i2       165        96        96  
//       i3        96         1         1  
//      off                   0     16032  
//      vec                   1         1  
// 
// Names: { i1, i2_i3 }
// Ranges: { 165, 15840 }
// Out stride: { 15840, 1 }
// Input 1 offset: 16032
// Input 1 stride: { 15936, 1 }
// Tile size: { 1, 512 }
// Contraction output var shape: fp32(1, 165, 165, 96):(2613600, 15840, 96, 1):10209.4 KiB
// Computed true ops: 5227200
// Computed work groups: 5115
// Computed inner loops: 1
// Computed shared mem: 2048
// Computed out regs: 2048
// Computed mem read: 2048
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7936, 165, 1
__kernel void kernel_c42_sdk_87(__global float* restrict  X_T272, __global const float* restrict  in1)
{
  in1 = (in1 + 16032);
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[512];
  int i2_i3_gid = (get_group_id(0) * 512);
  int i1_gid = get_group_id(1);
  {
    {
      int gbase = (i2_i3_gid + (i1_gid * 15936));
      int i2_i3_tid = (tid % 256);
      for (int i2_i3_lid = 0; i2_i3_lid < 2; i2_i3_lid += 1)
      {
        int i2_i3 = ((256 * i2_i3_lid) + i2_i3_tid);
        int gidx = (gbase + i2_i3);
        in1_shared[i2_i3] = in1[clamp((int)gidx, (int)-16032, (int)2629343)];
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
    int i2_i3_cond = ((i2_i3_lid < 1) || ((i2_i3_gid != 15360) || (i2_i3_tid < 224)));
    if (i2_i3_cond)
    {
      int i2_i3 = ((256 * i2_i3_lid) + i2_i3_tid);
      float LX_T272 = agg[i2_i3_lid];
      int gout_idx = ((15840 * i1_gid) + (i2_i3_gid + i2_i3));
      X_T272[gout_idx] = LX_T272;
    }
  }
}
