#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 28416 1 1
// lid: 256 1 1
// Original:
// X_T270[i0, i1, i2, i3 : _T380, _T381, _T382, _T383] = =(X_T269[i0, 1 + i1, 1 + i2, i3])
// With Index Variables Made Integral:
// X_T270[i0, i1, i2, i3 : _T380, _T381, _T382, _T383] = =(X_T269[i0, 1 + i1, 1 + i2, i3]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000, 500000000 + i2 < 1000000000, 500000000 + i3 < 1000000000
// Constraints:{ 0 <= i0 < 1, 0 <= i0 < 1, 0 <= i3 < 32, 0 <= i3 < 32, 0 <= i1 < 111, 0 <= i2 < 111, 0 <= 1 + i1 < 112, 0 <= 1 + i2 < 112, 0 <= 500000000 + i0 < 1000000000, 0 <= 500000000 + i1 < 1000000000, 0 <= 500000000 + i2 < 1000000000, 0 <= 500000000 + i3 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= i0 < 1, 0 <= i3 < 32, 0 <= i1 < 111, 0 <= i2 < 111 }
// Defracted:
// X_T270[i0, i1, i2, i3 : _T380, _T381, _T382, _T383] = =(X_T269[i0, 1 + i1, 1 + i2, i3]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000, 500000000 + i2 < 1000000000, 500000000 + i3 < 1000000000
// Flattened:
//              Range    X_T270    X_T269  
//       i1       111      3552      3584  
//       i2       111        32        32  
//       i3        32         1         1  
//      off                   0      3616  
//      vec                   1         1  
// 
// Names: { i1, i2_i3 }
// Ranges: { 111, 3552 }
// Out stride: { 3552, 1 }
// Input 1 offset: 3616
// Input 1 stride: { 3584, 1 }
// Tile size: { 1, 3552 }
// Contraction output var shape: fp32(1, 111, 111, 32):(394272, 3552, 32, 1):1540.12 KiB
// Computed true ops: 788544
// Computed work groups: 111
// Computed inner loops: 1
// Computed shared mem: 14208
// Computed out regs: 14336
// Computed mem read: 14208
// Computed mem write: 14208
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 28416, 1, 1
__kernel void kernel_c42_sdk_87(__global float* restrict  X_T270, __global const float* restrict  in1)
{
  in1 = (in1 + 3616);
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3552];
  int i1_gid = get_group_id(0);
  {
    {
      int gbase = (i1_gid * 3584);
      int i2_i3_tid = (tid % 256);
      for (int i2_i3_lid = 0; i2_i3_lid < 14; i2_i3_lid += 1)
      {
        int i2_i3_cond = ((i2_i3_lid < 13) || (i2_i3_tid < 224));
        if (i2_i3_cond)
        {
          int i2_i3 = ((256 * i2_i3_lid) + i2_i3_tid);
          int gidx = (gbase + i2_i3);
          in1_shared[i2_i3] = in1[clamp((int)gidx, (int)-3616, (int)397791)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int i2_i3_tid = (tid % 256);
    for (int i2_i3_lid = 0; i2_i3_lid < 14; i2_i3_lid += 1)
    {
      int i2_i3_cond = ((i2_i3_lid < 13) || (i2_i3_tid < 224));
      int i2_i3 = select((int)0, (int)((256 * i2_i3_lid) + i2_i3_tid), (int)i2_i3_cond);
      float val1 = in1_shared[i2_i3];
      agg[i2_i3_lid] = select((float)agg[i2_i3_lid], (float)val1, (int)i2_i3_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int i2_i3_tid = (tid % 256);
  for (int i2_i3_lid = 0; i2_i3_lid < 14; i2_i3_lid += 1)
  {
    int i2_i3_cond = ((i2_i3_lid < 13) || (i2_i3_tid < 224));
    if (i2_i3_cond)
    {
      int i2_i3 = ((256 * i2_i3_lid) + i2_i3_tid);
      float LX_T270 = agg[i2_i3_lid];
      int gout_idx = ((3552 * i1_gid) + i2_i3);
      X_T270[gout_idx] = LX_T270;
    }
  }
}
