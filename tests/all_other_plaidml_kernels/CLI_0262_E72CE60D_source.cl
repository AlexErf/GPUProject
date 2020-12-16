#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 21 1
// lid: 256 1 1
// Original:
// X_T3030[n, o0, o1, c : _T4824, _T4825, _T4826, _T4827] = =(X_T100[])
// With Index Variables Made Integral:
// X_T3030[n, o0, o1, c : _T4824, _T4825, _T4826, _T4827] = =(X_T100[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 21, 0 <= o1 < 21, 0 <= c < 2016, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 21, 0 <= o1 < 21, 0 <= c < 2016 }
// Defracted:
// X_T3030[n, o0, o1, c : _T4824, _T4825, _T4826, _T4827] = =(X_T100[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range   X_T3030    X_T100  
//        c      2016         1         0  
//       o0        21     42336         0  
//       o1        21      2016         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 2016, 21, 21 }
// Out stride: { 1, 42336, 2016 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 2016, 1, 1 }
// Contraction output var shape: fp32(1, 21, 21, 2016):(889056, 42336, 2016, 1):3472.88 KiB
// Computed true ops: 1778112
// Computed work groups: 441
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 8192
// Computed mem read: 128
// Computed mem write: 8064
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 21, 1
__kernel void kernel_c42_sdk_1173(__global float* restrict  X_T3030)
{
  int tid = get_local_id(0);
  float agg[8] = {0, 0, 0, 0, 0, 0, 0, 0, };
  int o1_gid = get_group_id(0);
  int o0_gid = get_group_id(1);
  {
    int c_tid = (tid % 256);
    for (int c_lid = 0; c_lid < 8; c_lid += 1)
    {
      int c_cond = ((c_lid < 7) || (c_tid < 224));
      int c = select((int)0, (int)((256 * c_lid) + c_tid), (int)c_cond);
      float val1 = 1.0f;
      agg[c_lid] = select((float)agg[c_lid], (float)val1, (int)c_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 256);
  for (int c_lid = 0; c_lid < 8; c_lid += 1)
  {
    int c_cond = ((c_lid < 7) || (c_tid < 224));
    if (c_cond)
    {
      int c = ((256 * c_lid) + c_tid);
      float LX_T3030 = agg[c_lid];
      int gout_idx = ((c + (42336 * (int)o0_gid)) + (2016 * o1_gid));
      X_T3030[gout_idx] = LX_T3030;
    }
  }
}
