#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 9 1
// lid: 256 1 1
// Original:
// X_T358[n0, n1, n2, a : _T470, _T471, _T472, _T473] = =(X_T357[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T358[n0, n1, n2, a : _T470, _T471, _T472, _T473] = =(X_T357[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= a < 288, 0 <= a < 320, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= a < 288 }
// Defracted:
// X_T358[n0, n1, n2, a : _T470, _T471, _T472, _T473] = =(X_T357[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T358    X_T357  
//        a       288         1         1  
//       n1        28      8960      8064  
//       n2        28       320       288  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 288, 28, 28 }
// Out stride: { 1, 8960, 320 }
// Input 1 offset: 0
// Input 1 stride: { 1, 8064, 288 }
// Tile size: { 32, 4, 28 }
// Contraction output var shape: fp32(1, 28, 28, 320):(250880, 8960, 320, 1):980 KiB
// Computed true ops: 451584
// Computed work groups: 63
// Computed inner loops: 1
// Computed shared mem: 14464
// Computed out regs: 14336
// Computed mem read: 14336
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 9, 1
__kernel void kernel_c68_sdk_111(__global float* restrict  X_T358, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3616];
  int a_gid = (get_group_id(1) * 32);
  int n1_gid = (get_group_id(0) * 4);
  {
    {
      int gbase = (a_gid + (n1_gid * 8064));
      int a_tid = (tid % 32);
      int n2_n1_tid = ((tid / 32) % 8);
      for (int n2_n1_lid = 0; n2_n1_lid < 14; n2_n1_lid += 1)
      {
        int n2_n1 = ((8 * n2_n1_lid) + n2_n1_tid);
        int lidx = ((113 * a_tid) + n2_n1);
        int gidx = ((gbase + a_tid) + (288 * n2_n1));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)225791)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 4);
    int n1_tid = ((tid / 128) % 2);
    for (int n2_lid = 0; n2_lid < 7; n2_lid += 1)
    {
      int n2 = ((4 * n2_lid) + n2_tid);
      for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
      {
        int n1 = ((2 * n1_lid) + n1_tid);
        float val1 = in1_shared[(((113 * a_tid) + n2) + (28 * n1))];
        int agg_idx = (n2_lid + (n1_lid * 7));
        agg[agg_idx] = val1;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 4);
  int n1_tid = ((tid / 128) % 2);
  for (int n2_lid = 0; n2_lid < 7; n2_lid += 1)
  {
    int n2 = ((4 * n2_lid) + n2_tid);
    for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
    {
      int n1 = ((2 * n1_lid) + n1_tid);
      float LX_T358 = agg[(n2_lid + (n1_lid * 7))];
      int gout_idx = (((a_gid + a_tid) + (8960 * (n1_gid + n1))) + (320 * n2));
      X_T358[gout_idx] = LX_T358;
    }
  }
}
