#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2304 35 1
// lid: 256 1 1
// Original:
// X_T323[n, o0, o1, c : _T433, _T434, _T435, _T436] = =(X_T160[])
// With Index Variables Made Integral:
// X_T323[n, o0, o1, c : _T433, _T434, _T435, _T436] = =(X_T160[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 35, 0 <= o1 < 35, 0 <= c < 288, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 35, 0 <= o1 < 35, 0 <= c < 288 }
// Defracted:
// X_T323[n, o0, o1, c : _T433, _T434, _T435, _T436] = =(X_T160[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range    X_T323    X_T160  
//        c       288         1         0  
//       o0        35     10080         0  
//       o1        35       288         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 288, 35, 35 }
// Out stride: { 1, 10080, 288 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 32, 35, 1 }
// Contraction output var shape: fp32(1, 35, 35, 288):(352800, 10080, 288, 1):1378.12 KiB
// Computed true ops: 705600
// Computed work groups: 315
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 5120
// Computed mem read: 128
// Computed mem write: 4480
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2304, 35, 1
__kernel void kernel_c56_sdk_103(__global float* restrict  X_T323)
{
  int tid = get_local_id(0);
  float agg[5] = {0, 0, 0, 0, 0, };
  int c_gid = (get_group_id(0) * 32);
  int o1_gid = get_group_id(1);
  {
    int c_tid = (tid % 32);
    int o0_tid = ((tid / 32) % 8);
    for (int o0_lid = 0; o0_lid < 5; o0_lid += 1)
    {
      int o0_cond = ((o0_lid < 4) || (o0_tid < 3));
      int o0 = select((int)0, (int)((8 * o0_lid) + o0_tid), (int)o0_cond);
      float val1 = 1.0f;
      agg[o0_lid] = select((float)agg[o0_lid], (float)val1, (int)o0_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o0_tid = ((tid / 32) % 8);
  for (int o0_lid = 0; o0_lid < 5; o0_lid += 1)
  {
    int o0_cond = ((o0_lid < 4) || (o0_tid < 3));
    if (o0_cond)
    {
      int o0 = ((8 * o0_lid) + o0_tid);
      float LX_T323 = agg[o0_lid];
      int gout_idx = (((c_gid + c_tid) + (10080 * o0)) + (288 * o1_gid));
      X_T323[gout_idx] = LX_T323;
    }
  }
}
