#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T1615[n0, n1, n2, 440 + a : _T2552, _T2553, _T2554, _T2555] = =(X_T1614[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T1615[n0, n1, n2, 440 + a : _T2552, _T2553, _T2554, _T2555] = =(X_T1614[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 88, 0 <= 440 + a < 528, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 88 }
// Defracted:
// X_T1615[n0, n1, n2, 440 + a : _T2552, _T2553, _T2554, _T2555] = =(X_T1614[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T1615   X_T1614  
//        a        88         1         1  
//       n1        14      7392      1232  
//       n2        14       528        88  
//      off                 440         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 88, 14, 14 }
// Out stride: { 1, 7392, 528 }
// Input 1 offset: 0
// Input 1 stride: { 1, 1232, 88 }
// Tile size: { 88, 2, 2 }
// Contraction output var shape: fp32(1, 14, 14, 528):(103488, 7392, 528, 1):404.25 KiB
// Computed true ops: 34496
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 1416
// Computed out regs: 2048
// Computed mem read: 1408
// Computed mem write: 1536
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c42_sdk_613(__global float* restrict  X_T1615, __global const float* restrict  in1)
{
  X_T1615 = (X_T1615 + 440);
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[354];
  int n2_gid = (get_group_id(0) * 2);
  int n1_gid = (get_group_id(1) * 2);
  {
    {
      int gbase = ((n2_gid * 88) + (n1_gid * 1232));
      int a_n2_tid = (tid % 256);
      int a_n2_cond = (a_n2_tid < 176);
      if (a_n2_cond)
      {
        for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
        {
          int lidx = (a_n2_tid + (177 * n1_lid));
          int gidx = ((gbase + a_n2_tid) + (1232 * n1_lid));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)17247)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 64);
    int n2_tid = ((tid / 64) % 2);
    int n1_tid = ((tid / 128) % 2);
    for (int a_lid = 0; a_lid < 2; a_lid += 1)
    {
      int a_cond = ((a_lid < 1) || (a_tid < 24));
      int a = select((int)0, (int)((64 * a_lid) + a_tid), (int)a_cond);
      float val1 = in1_shared[((a + (88 * n2_tid)) + (177 * n1_tid))];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)a_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 64);
  int n2_tid = ((tid / 64) % 2);
  int n1_tid = ((tid / 128) % 2);
  for (int a_lid = 0; a_lid < 2; a_lid += 1)
  {
    int a_cond = ((a_lid < 1) || (a_tid < 24));
    if (a_cond)
    {
      int a = ((64 * a_lid) + a_tid);
      float LX_T1615 = agg[a_lid];
      int gout_idx = ((a + (7392 * (n1_gid + n1_tid))) + (528 * (n2_gid + n2_tid)));
      X_T1615[gout_idx] = LX_T1615;
    }
  }
}
