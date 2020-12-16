#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5376 42 1
// lid: 256 1 1
// Original:
// X_T470[n0, n1, n2, a : _T706, _T707, _T708, _T709] = =(X_T469[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T470[n0, n1, n2, a : _T706, _T707, _T708, _T709] = =(X_T469[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 42, 0 <= n2 < 42, 0 <= n1 < 42, 0 <= n2 < 42, 0 <= a < 168, 0 <= a < 1008, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 42, 0 <= n2 < 42, 0 <= a < 168 }
// Defracted:
// X_T470[n0, n1, n2, a : _T706, _T707, _T708, _T709] = =(X_T469[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T470    X_T469  
//        a       168         1         1  
//       n1        42     42336      7056  
//       n2        42      1008       168  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 168, 42, 42 }
// Out stride: { 1, 42336, 1008 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7056, 168 }
// Tile size: { 168, 2, 1 }
// Contraction output var shape: fp32(1, 42, 42, 1008):(1778112, 42336, 1008, 1):6945.75 KiB
// Computed true ops: 592704
// Computed work groups: 882
// Computed inner loops: 1
// Computed shared mem: 1352
// Computed out regs: 2048
// Computed mem read: 1280
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5376, 42, 1
__kernel void kernel_c42_sdk_163(__global float* restrict  X_T470, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[338];
  int n2_gid = get_group_id(1);
  int n1_gid = (get_group_id(0) * 2);
  {
    {
      int gbase = ((n2_gid * 168) + (n1_gid * 7056));
      int a_n2_tid = (tid % 256);
      int a_n2_cond = (a_n2_tid < 168);
      if (a_n2_cond)
      {
        for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
        {
          int lidx = (a_n2_tid + (169 * n1_lid));
          int gidx = ((gbase + a_n2_tid) + (7056 * n1_lid));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)296351)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 128);
    int n1_tid = ((tid / 128) % 2);
    for (int a_lid = 0; a_lid < 2; a_lid += 1)
    {
      int a_cond = ((a_lid < 1) || (a_tid < 40));
      int a = select((int)0, (int)((128 * a_lid) + a_tid), (int)a_cond);
      float val1 = in1_shared[(a + (169 * n1_tid))];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)a_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 128);
  int n1_tid = ((tid / 128) % 2);
  for (int a_lid = 0; a_lid < 2; a_lid += 1)
  {
    int a_cond = ((a_lid < 1) || (a_tid < 40));
    if (a_cond)
    {
      int a = ((128 * a_lid) + a_tid);
      float LX_T470 = agg[a_lid];
      int gout_idx = ((a + (42336 * (int)(n1_gid + n1_tid))) + (1008 * n2_gid));
      X_T470[gout_idx] = LX_T470;
    }
  }
}
