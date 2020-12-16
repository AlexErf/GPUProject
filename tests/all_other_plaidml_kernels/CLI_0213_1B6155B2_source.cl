#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 331 1
// lid: 256 1 1
// Original:
// X_T1774[i0, i1, i2, i3 : _T2801, _T2802, _T2803, _T2804] = =(X_T1773[i0, 1 + i1, 1 + i2, i3])
// With Index Variables Made Integral:
// X_T1774[i0, i1, i2, i3 : _T2801, _T2802, _T2803, _T2804] = =(X_T1773[i0, 1 + i1, 1 + i2, i3]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000, 500000000 + i2 < 1000000000, 500000000 + i3 < 1000000000
// Constraints:{ 0 <= i0 < 1, 0 <= i0 < 1, 0 <= i1 < 42, 0 <= i2 < 42, 0 <= 1 + i1 < 43, 0 <= 1 + i2 < 43, 0 <= i3 < 1008, 0 <= i3 < 1008, 0 <= 500000000 + i0 < 1000000000, 0 <= 500000000 + i1 < 1000000000, 0 <= 500000000 + i2 < 1000000000, 0 <= 500000000 + i3 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= i0 < 1, 0 <= i1 < 42, 0 <= i2 < 42, 0 <= i3 < 1008 }
// Defracted:
// X_T1774[i0, i1, i2, i3 : _T2801, _T2802, _T2803, _T2804] = =(X_T1773[i0, 1 + i1, 1 + i2, i3]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000, 500000000 + i2 < 1000000000, 500000000 + i3 < 1000000000
// Flattened:
//              Range   X_T1774   X_T1773  
//       i1        42     42336     43344  
//       i2        42      1008      1008  
//       i3      1008         1         1  
//      off                   0     44352  
//      vec                   1         1  
// 
// Names: { i1, i2_i3 }
// Ranges: { 42, 42336 }
// Out stride: { 42336, 1 }
// Input 1 offset: 44352
// Input 1 stride: { 43344, 1 }
// Tile size: { 2, 128 }
// Contraction output var shape: fp32(1, 42, 42, 1008):(1778112, 42336, 1008, 1):6945.75 KiB
// Computed true ops: 3556224
// Computed work groups: 6951
// Computed inner loops: 1
// Computed shared mem: 1032
// Computed out regs: 1024
// Computed mem read: 1024
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 331, 1
__kernel void kernel_c42_sdk_676(__global float* restrict  X_T1774, __global const float* restrict  in1)
{
  in1 = (in1 + 44352);
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[258];
  int i2_i3_gid = (get_group_id(1) * 128);
  int i1_gid = (get_group_id(0) * 2);
  {
    {
      int gbase = (i2_i3_gid + (i1_gid * 43344));
      int i2_i3_tid = (tid % 128);
      int i1_tid = ((tid / 128) % 2);
      int lidx = (i2_i3_tid + (129 * i1_tid));
      int gidx = ((gbase + i2_i3_tid) + (43344 * (int)i1_tid));
      in1_shared[lidx] = in1[clamp((int)gidx, (int)-44352, (int)1819439)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int i2_i3_tid = (tid % 128);
    int i1_tid = ((tid / 128) % 2);
    float val1 = in1_shared[(i2_i3_tid + (129 * i1_tid))];
    agg[0] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int i2_i3_tid = (tid % 128);
  int i1_tid = ((tid / 128) % 2);
  int i2_i3_cond = ((i2_i3_gid != 42240) || (i2_i3_tid < 96));
  if (i2_i3_cond)
  {
    float LX_T1774 = agg[0];
    int gout_idx = ((42336 * (int)(i1_gid + i1_tid)) + (int)(i2_i3_gid + i2_i3_tid));
    X_T1774[gout_idx] = LX_T1774;
  }
}
