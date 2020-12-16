#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T1618[n0, n1, n2, a : _T2308, _T2309, _T2310, _T2311] = =(X_T1617[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T1618[n0, n1, n2, a : _T2308, _T2309, _T2310, _T2311] = =(X_T1617[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 928, 0 <= a < 960, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 928 }
// Defracted:
// X_T1618[n0, n1, n2, a : _T2308, _T2309, _T2310, _T2311] = =(X_T1617[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T1618   X_T1617  
//        a       928         1         1  
//       n1         7      6720      6496  
//       n2         7       960       928  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 928, 7, 7 }
// Out stride: { 1, 6720, 960 }
// Input 1 offset: 0
// Input 1 stride: { 1, 6496, 928 }
// Tile size: { 928, 1, 1 }
// Contraction output var shape: fp32(1, 7, 7, 960):(47040, 6720, 960, 1):183.75 KiB
// Computed true ops: 90944
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 3712
// Computed out regs: 4096
// Computed mem read: 3712
// Computed mem write: 3712
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c108_sdk_555(__global float* restrict  X_T1618, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[928];
  int n2_gid = get_group_id(0);
  int n1_gid = get_group_id(1);
  {
    {
      int gbase = ((n2_gid * 928) + (n1_gid * 6496));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 4; a_n2_lid += 1)
      {
        int a_n2_cond = ((a_n2_lid < 3) || (a_n2_tid < 160));
        if (a_n2_cond)
        {
          int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
          int gidx = (gbase + a_n2);
          in1_shared[a_n2] = in1[clamp((int)gidx, (int)0, (int)45471)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 256);
    for (int a_lid = 0; a_lid < 4; a_lid += 1)
    {
      int a_cond = ((a_lid < 3) || (a_tid < 160));
      int a = select((int)0, (int)((256 * a_lid) + a_tid), (int)a_cond);
      float val1 = in1_shared[a];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)a_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 256);
  for (int a_lid = 0; a_lid < 4; a_lid += 1)
  {
    int a_cond = ((a_lid < 3) || (a_tid < 160));
    if (a_cond)
    {
      int a = ((256 * a_lid) + a_tid);
      float LX_T1618 = agg[a_lid];
      int gout_idx = ((a + (6720 * n1_gid)) + (960 * n2_gid));
      X_T1618[gout_idx] = LX_T1618;
    }
  }
}
