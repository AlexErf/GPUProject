#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Original:
// X_T199[n, o0, o1, c : _T258, _T259, _T260, _T261] = =(X_T97[])
// With Index Variables Made Integral:
// X_T199[n, o0, o1, c : _T258, _T259, _T260, _T261] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= c < 11, 0 <= o0 < 56, 0 <= o1 < 56, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= c < 11, 0 <= o0 < 56, 0 <= o1 < 56 }
// Defracted:
// X_T199[n, o0, o1, c : _T258, _T259, _T260, _T261] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range    X_T199     X_T97  
//        c        11         1         0  
//       o0        56       616         0  
//       o1        56        11         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 11, 56, 56 }
// Out stride: { 1, 616, 11 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 11, 56, 2 }
// Contraction output var shape: fp32(1, 56, 56, 11):(34496, 616, 11, 1):134.75 KiB
// Computed true ops: 68992
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 7168
// Computed mem read: 128
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c42_sdk_58(__global float* restrict  X_T199)
{
  int tid = get_local_id(0);
  float agg[7] = {0, 0, 0, 0, 0, 0, 0, };
  int o1_gid = (get_group_id(0) * 2);
  {
    int c_tid = (tid % 16);
    int o1_tid = ((tid / 16) % 2);
    int o0_tid = ((tid / 32) % 8);
    int c_cond = (c_tid < 11);
    int c = select((int)0, (int)c_tid, (int)c_cond);
    for (int o0_lid = 0; o0_lid < 7; o0_lid += 1)
    {
      int o0 = ((8 * o0_lid) + o0_tid);
      float val1 = 1.0f;
      agg[o0_lid] = select((float)agg[o0_lid], (float)val1, (int)c_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 16);
  int o1_tid = ((tid / 16) % 2);
  int o0_tid = ((tid / 32) % 8);
  int c_cond = (c_tid < 11);
  if (c_cond)
  {
    for (int o0_lid = 0; o0_lid < 7; o0_lid += 1)
    {
      int o0 = ((8 * o0_lid) + o0_tid);
      float LX_T199 = agg[o0_lid];
      int gout_idx = ((c_tid + (616 * o0)) + (11 * (o1_gid + o1_tid)));
      X_T199[gout_idx] = LX_T199;
    }
  }
}
