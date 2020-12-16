#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 59136 1 1
// lid: 256 1 1
// Original:
// X_T1428[i0, i1, i2, i3 : _T2245, _T2246, _T2247, _T2248] = =(X_T1427[i0, 1 + i1, 1 + i2, i3])
// With Index Variables Made Integral:
// X_T1428[i0, i1, i2, i3 : _T2245, _T2246, _T2247, _T2248] = =(X_T1427[i0, 1 + i1, 1 + i2, i3]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000, 500000000 + i2 < 1000000000, 500000000 + i3 < 1000000000
// Constraints:{ 0 <= i0 < 1, 0 <= i0 < 1, 0 <= i1 < 28, 0 <= i2 < 28, 0 <= 1 + i1 < 29, 0 <= 1 + i2 < 29, 0 <= i3 < 264, 0 <= i3 < 264, 0 <= 500000000 + i0 < 1000000000, 0 <= 500000000 + i1 < 1000000000, 0 <= 500000000 + i2 < 1000000000, 0 <= 500000000 + i3 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= i0 < 1, 0 <= i1 < 28, 0 <= i2 < 28, 0 <= i3 < 264 }
// Defracted:
// X_T1428[i0, i1, i2, i3 : _T2245, _T2246, _T2247, _T2248] = =(X_T1427[i0, 1 + i1, 1 + i2, i3]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000, 500000000 + i2 < 1000000000, 500000000 + i3 < 1000000000
// Flattened:
//              Range   X_T1428   X_T1427  
//       i1        28      7392      7656  
//       i2        28       264       264  
//       i3       264         1         1  
//      off                   0      7920  
//      vec                   1         1  
// 
// Names: { i1, i2_i3 }
// Ranges: { 28, 7392 }
// Out stride: { 7392, 1 }
// Input 1 offset: 7920
// Input 1 stride: { 7656, 1 }
// Tile size: { 28, 32 }
// Contraction output var shape: fp32(1, 28, 28, 264):(206976, 7392, 264, 1):808.5 KiB
// Computed true ops: 413952
// Computed work groups: 231
// Computed inner loops: 1
// Computed shared mem: 3696
// Computed out regs: 4096
// Computed mem read: 3584
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 59136, 1, 1
__kernel void kernel_c42_sdk_538(__global float* restrict  X_T1428, __global const float* restrict  in1)
{
  in1 = (in1 + 7920);
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[924];
  int i2_i3_gid = (get_group_id(0) * 32);
  {
    {
      int i2_i3_tid = (tid % 32);
      int i1_tid = ((tid / 32) % 8);
      for (int i1_lid = 0; i1_lid < 4; i1_lid += 1)
      {
        int i1_cond = ((i1_lid < 3) || (i1_tid < 4));
        if (i1_cond)
        {
          int i1 = ((8 * i1_lid) + i1_tid);
          int lidx = (i2_i3_tid + (33 * i1));
          int gidx = ((i2_i3_gid + i2_i3_tid) + (7656 * i1));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)-7920, (int)214103)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int i2_i3_tid = (tid % 32);
    int i1_tid = ((tid / 32) % 8);
    for (int i1_lid = 0; i1_lid < 4; i1_lid += 1)
    {
      int i1_cond = ((i1_lid < 3) || (i1_tid < 4));
      int i1 = select((int)0, (int)((8 * i1_lid) + i1_tid), (int)i1_cond);
      float val1 = in1_shared[(i2_i3_tid + (33 * i1))];
      agg[i1_lid] = select((float)agg[i1_lid], (float)val1, (int)i1_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int i2_i3_tid = (tid % 32);
  int i1_tid = ((tid / 32) % 8);
  for (int i1_lid = 0; i1_lid < 4; i1_lid += 1)
  {
    int i1_cond = ((i1_lid < 3) || (i1_tid < 4));
    if (i1_cond)
    {
      int i1 = ((8 * i1_lid) + i1_tid);
      float LX_T1428 = agg[i1_lid];
      int gout_idx = ((7392 * i1) + (i2_i3_gid + i2_i3_tid));
      X_T1428[gout_idx] = LX_T1428;
    }
  }
}
