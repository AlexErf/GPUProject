#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 19712 1 1
// lid: 256 1 1
// Original:
// X_T494[i0, i1, i2, i3 : _T743, _T744, _T745, _T746] = =(X_T493[i0, 1 + i1, 1 + i2, i3])
// With Index Variables Made Integral:
// X_T494[i0, i1, i2, i3 : _T743, _T744, _T745, _T746] = =(X_T493[i0, 1 + i1, 1 + i2, i3]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000, 500000000 + i2 < 1000000000, 500000000 + i3 < 1000000000
// Constraints:{ 0 <= i0 < 1, 0 <= i0 < 1, 0 <= i3 < 44, 0 <= i3 < 44, 0 <= i1 < 56, 0 <= i2 < 56, 0 <= 1 + i1 < 57, 0 <= 1 + i2 < 57, 0 <= 500000000 + i0 < 1000000000, 0 <= 500000000 + i1 < 1000000000, 0 <= 500000000 + i2 < 1000000000, 0 <= 500000000 + i3 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= i0 < 1, 0 <= i3 < 44, 0 <= i1 < 56, 0 <= i2 < 56 }
// Defracted:
// X_T494[i0, i1, i2, i3 : _T743, _T744, _T745, _T746] = =(X_T493[i0, 1 + i1, 1 + i2, i3]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000, 500000000 + i2 < 1000000000, 500000000 + i3 < 1000000000
// Flattened:
//              Range    X_T494    X_T493  
//       i1        56      2464      2508  
//       i2        56        44        44  
//       i3        44         1         1  
//      off                   0      2552  
//      vec                   1         1  
// 
// Names: { i1, i2_i3 }
// Ranges: { 56, 2464 }
// Out stride: { 2464, 1 }
// Input 1 offset: 2552
// Input 1 stride: { 2508, 1 }
// Tile size: { 56, 32 }
// Contraction output var shape: fp32(1, 56, 56, 44):(137984, 2464, 44, 1):539 KiB
// Computed true ops: 275968
// Computed work groups: 77
// Computed inner loops: 1
// Computed shared mem: 7296
// Computed out regs: 7168
// Computed mem read: 7168
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 19712, 1, 1
__kernel void kernel_c42_sdk_172(__global float* restrict  X_T494, __global const float* restrict  in1)
{
  in1 = (in1 + 2552);
  int tid = get_local_id(0);
  float agg[7] = {0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[1824];
  int i2_i3_gid = (get_group_id(0) * 32);
  {
    {
      int i2_i3_tid = (tid % 32);
      int i1_tid = ((tid / 32) % 8);
      for (int i1_lid = 0; i1_lid < 7; i1_lid += 1)
      {
        int i1 = ((8 * i1_lid) + i1_tid);
        int lidx = ((57 * i2_i3_tid) + i1);
        int gidx = ((i2_i3_gid + i2_i3_tid) + (2508 * i1));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)-2552, (int)140403)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int i2_i3_tid = (tid % 32);
    int i1_tid = ((tid / 32) % 8);
    for (int i1_lid = 0; i1_lid < 7; i1_lid += 1)
    {
      int i1 = ((8 * i1_lid) + i1_tid);
      float val1 = in1_shared[((57 * i2_i3_tid) + i1)];
      agg[i1_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int i2_i3_tid = (tid % 32);
  int i1_tid = ((tid / 32) % 8);
  for (int i1_lid = 0; i1_lid < 7; i1_lid += 1)
  {
    int i1 = ((8 * i1_lid) + i1_tid);
    float LX_T494 = agg[i1_lid];
    int gout_idx = ((2464 * i1) + (i2_i3_gid + i2_i3_tid));
    X_T494[gout_idx] = LX_T494;
  }
}
