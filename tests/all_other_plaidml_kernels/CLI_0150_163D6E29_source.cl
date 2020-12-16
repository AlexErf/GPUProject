#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 14080 83 1
// lid: 256 1 1
// Original:
// X_T493[i0, i1, i2, i3 : _T743, _T744, _T745, _T746] = =(X_T492[i0, 1 + i1, 1 + i2, i3])
// With Index Variables Made Integral:
// X_T493[i0, i1, i2, i3 : _T743, _T744, _T745, _T746] = =(X_T492[i0, 1 + i1, 1 + i2, i3]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000, 500000000 + i2 < 1000000000, 500000000 + i3 < 1000000000
// Constraints:{ 0 <= i0 < 1, 0 <= i0 < 1, 0 <= i1 < 83, 0 <= i2 < 83, 0 <= 1 + i1 < 84, 0 <= 1 + i2 < 84, 0 <= i3 < 168, 0 <= i3 < 168, 0 <= 500000000 + i0 < 1000000000, 0 <= 500000000 + i1 < 1000000000, 0 <= 500000000 + i2 < 1000000000, 0 <= 500000000 + i3 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= i0 < 1, 0 <= i1 < 83, 0 <= i2 < 83, 0 <= i3 < 168 }
// Defracted:
// X_T493[i0, i1, i2, i3 : _T743, _T744, _T745, _T746] = =(X_T492[i0, 1 + i1, 1 + i2, i3]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000, 500000000 + i2 < 1000000000, 500000000 + i3 < 1000000000
// Flattened:
//              Range    X_T493    X_T492  
//       i1        83     13944     14112  
//       i2        83       168       168  
//       i3       168         1         1  
//      off                   0     14280  
//      vec                   1         1  
// 
// Names: { i1, i2_i3 }
// Ranges: { 83, 13944 }
// Out stride: { 13944, 1 }
// Input 1 offset: 14280
// Input 1 stride: { 14112, 1 }
// Tile size: { 1, 256 }
// Contraction output var shape: fp32(1, 83, 83, 168):(1157352, 13944, 168, 1):4520.91 KiB
// Computed true ops: 2314704
// Computed work groups: 4565
// Computed inner loops: 1
// Computed shared mem: 1024
// Computed out regs: 1024
// Computed mem read: 1024
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 14080, 83, 1
__kernel void kernel_c42_sdk_172(__global float* restrict  X_T493, __global const float* restrict  in1)
{
  in1 = (in1 + 14280);
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[256];
  int i2_i3_gid = (get_group_id(0) * 256);
  int i1_gid = get_group_id(1);
  {
    {
      int gbase = (i2_i3_gid + (i1_gid * 14112));
      int i2_i3_tid = (tid % 256);
      int gidx = (gbase + i2_i3_tid);
      in1_shared[i2_i3_tid] = in1[clamp((int)gidx, (int)-14280, (int)1171127)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int i2_i3_tid = (tid % 256);
    float val1 = in1_shared[i2_i3_tid];
    agg[0] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int i2_i3_tid = (tid % 256);
  int i2_i3_cond = ((i2_i3_gid != 13824) || (i2_i3_tid < 120));
  if (i2_i3_cond)
  {
    float LX_T493 = agg[0];
    int gout_idx = ((13944 * i1_gid) + (i2_i3_gid + i2_i3_tid));
    X_T493[gout_idx] = LX_T493;
  }
}
