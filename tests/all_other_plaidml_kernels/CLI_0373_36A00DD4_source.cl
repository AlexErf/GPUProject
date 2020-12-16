#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 14 1
// lid: 256 1 1
// Original:
// X_T881[n0, n1, n2, a : _T1204, _T1205, _T1206, _T1207] = =(X_T880[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T881[n0, n1, n2, a : _T1204, _T1205, _T1206, _T1207] = =(X_T880[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 640, 0 <= a < 672, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 640 }
// Defracted:
// X_T881[n0, n1, n2, a : _T1204, _T1205, _T1206, _T1207] = =(X_T880[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T881    X_T880  
//        a       640         1         1  
//       n1        14      9408      8960  
//       n2        14       672       640  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 640, 14, 14 }
// Out stride: { 1, 9408, 672 }
// Input 1 offset: 0
// Input 1 stride: { 1, 8960, 640 }
// Tile size: { 640, 1, 1 }
// Contraction output var shape: fp32(1, 14, 14, 672):(131712, 9408, 672, 1):514.5 KiB
// Computed true ops: 250880
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 2560
// Computed out regs: 3072
// Computed mem read: 2560
// Computed mem write: 2560
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 14, 1
__kernel void kernel_c124_sdk_288(__global float* restrict  X_T881, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[3] = {0, 0, 0, };
  __local float in1_shared[640];
  int n2_gid = get_group_id(0);
  int n1_gid = get_group_id(1);
  {
    {
      int gbase = ((n2_gid * 640) + (n1_gid * 8960));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 3; a_n2_lid += 1)
      {
        int a_n2_cond = ((a_n2_lid < 2) || (a_n2_tid < 128));
        if (a_n2_cond)
        {
          int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
          int gidx = (gbase + a_n2);
          in1_shared[a_n2] = in1[clamp((int)gidx, (int)0, (int)125439)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 256);
    for (int a_lid = 0; a_lid < 3; a_lid += 1)
    {
      int a_cond = ((a_lid < 2) || (a_tid < 128));
      int a = select((int)0, (int)((256 * a_lid) + a_tid), (int)a_cond);
      float val1 = in1_shared[a];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)a_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 256);
  for (int a_lid = 0; a_lid < 3; a_lid += 1)
  {
    int a_cond = ((a_lid < 2) || (a_tid < 128));
    if (a_cond)
    {
      int a = ((256 * a_lid) + a_tid);
      float LX_T881 = agg[a_lid];
      int gout_idx = ((a + (9408 * n1_gid)) + (672 * n2_gid));
      X_T881[gout_idx] = LX_T881;
    }
  }
}
