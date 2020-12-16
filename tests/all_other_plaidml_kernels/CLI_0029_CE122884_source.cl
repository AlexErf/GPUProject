#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 21 1
// lid: 256 1 1
// Original:
// X_T32[n, d0, d1, c : _T2, _T3, _T4, _T5] = =(X_I_247[n, -3 + d0, -3 + d1, c])
// With Index Variables Made Integral:
// X_T32[n, d0, d1, c : _T2, _T3, _T4, _T5] = =(X_I_247[n, -3 + d0, -3 + d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= c < 3, 0 <= c < 3, 0 <= -3 + d0 < 224, 0 <= -3 + d1 < 224, 0 <= d0 < 230, 0 <= d1 < 230, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= c < 3, 0 <= -3 + d0 < 224, 0 <= -3 + d1 < 224 }
// Defracted:
// X_T32[n, d0, d1, c : _T2, _T3, _T4, _T5] = =(X_I_247[n, -3 + d0, -3 + d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range     X_T32   X_I_247  
//        c         3         1         1  
//       d0       224       690       672  
//       d1       224         3         3  
//      off                2079         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 672, 224 }
// Out stride: { 1, 690 }
// Input 1 offset: 0
// Input 1 stride: { 1, 672 }
// Tile size: { 32, 32 }
// Contraction output var shape: fp32(1, 230, 230, 3):(158700, 690, 3, 1):619.922 KiB
// Computed true ops: 301056
// Computed work groups: 147
// Computed inner loops: 1
// Computed shared mem: 4224
// Computed out regs: 4096
// Computed mem read: 4096
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 21, 1
__kernel void kernel_c29_sdk_0(__global float* restrict  X_T32, __global const float* restrict  in1)
{
  X_T32 = (X_T32 + 2079);
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[1056];
  int d1_c_gid = (get_group_id(1) * 32);
  int d0_gid = (get_group_id(0) * 32);
  {
    {
      int gbase = (d1_c_gid + (d0_gid * 672));
      int d1_c_tid = (tid % 32);
      int d0_tid = ((tid / 32) % 8);
      for (int d0_lid = 0; d0_lid < 4; d0_lid += 1)
      {
        int d0 = ((8 * d0_lid) + d0_tid);
        int lidx = (d1_c_tid + (33 * d0));
        int gidx = ((gbase + d1_c_tid) + (672 * d0));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)150527)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int d1_c_tid = (tid % 32);
    int d0_tid = ((tid / 32) % 8);
    for (int d0_lid = 0; d0_lid < 4; d0_lid += 1)
    {
      int d0 = ((8 * d0_lid) + d0_tid);
      float val1 = in1_shared[(d1_c_tid + (33 * d0))];
      agg[d0_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int d1_c_tid = (tid % 32);
  int d0_tid = ((tid / 32) % 8);
  for (int d0_lid = 0; d0_lid < 4; d0_lid += 1)
  {
    int d0 = ((8 * d0_lid) + d0_tid);
    float LX_T32 = agg[d0_lid];
    int gout_idx = ((d1_c_gid + d1_c_tid) + (690 * (d0_gid + d0)));
    X_T32[gout_idx] = LX_T32;
  }
}
