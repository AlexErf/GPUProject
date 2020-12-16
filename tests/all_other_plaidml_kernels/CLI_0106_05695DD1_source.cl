#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 28416 1 1
// lid: 256 1 1
// Original:
// X_T258[n, o0, o1, c : _T356, _T357, _T358, _T359] = =(X_T97[])
// With Index Variables Made Integral:
// X_T258[n, o0, o1, c : _T356, _T357, _T358, _T359] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= c < 32, 0 <= o0 < 111, 0 <= o1 < 111, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= c < 32, 0 <= o0 < 111, 0 <= o1 < 111 }
// Defracted:
// X_T258[n, o0, o1, c : _T356, _T357, _T358, _T359] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range    X_T258     X_T97  
//        c        32         1         0  
//       o0       111      3552         0  
//       o1       111        32         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 32, 111, 111 }
// Out stride: { 1, 3552, 32 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 32, 111, 1 }
// Contraction output var shape: fp32(1, 111, 111, 32):(394272, 3552, 32, 1):1540.12 KiB
// Computed true ops: 788544
// Computed work groups: 111
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 14336
// Computed mem read: 128
// Computed mem write: 14208
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 28416, 1, 1
__kernel void kernel_c42_sdk_82(__global float* restrict  X_T258)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  int o1_gid = get_group_id(0);
  {
    int c_tid = (tid % 32);
    int o0_tid = ((tid / 32) % 8);
    for (int o0_lid = 0; o0_lid < 14; o0_lid += 1)
    {
      int o0_cond = ((o0_lid < 13) || (o0_tid < 7));
      int o0 = select((int)0, (int)((8 * o0_lid) + o0_tid), (int)o0_cond);
      float val1 = 1.0f;
      agg[o0_lid] = select((float)agg[o0_lid], (float)val1, (int)o0_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o0_tid = ((tid / 32) % 8);
  for (int o0_lid = 0; o0_lid < 14; o0_lid += 1)
  {
    int o0_cond = ((o0_lid < 13) || (o0_tid < 7));
    if (o0_cond)
    {
      int o0 = ((8 * o0_lid) + o0_tid);
      float LX_T258 = agg[o0_lid];
      int gout_idx = ((c_tid + (3552 * o0)) + (32 * o1_gid));
      X_T258[gout_idx] = LX_T258;
    }
  }
}
