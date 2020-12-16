#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 8 1
// lid: 256 1 1
// Original:
// X_T1999[n0, n1, n2, 992 + a : _T2803, _T2804, _T2805, _T2806] = =(X_T1998[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T1999[n0, n1, n2, 992 + a : _T2803, _T2804, _T2805, _T2806] = =(X_T1998[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 8, 0 <= n2 < 8, 0 <= n1 < 8, 0 <= n2 < 8, 0 <= a < 1088, 0 <= 992 + a < 2080, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 8, 0 <= n2 < 8, 0 <= a < 1088 }
// Defracted:
// X_T1999[n0, n1, n2, 992 + a : _T2803, _T2804, _T2805, _T2806] = =(X_T1998[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T1999   X_T1998  
//        a      1088         1         1  
//       n1         8     16640      8704  
//       n2         8      2080      1088  
//      off                 992         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 1088, 8, 8 }
// Out stride: { 1, 16640, 2080 }
// Input 1 offset: 0
// Input 1 stride: { 1, 8704, 1088 }
// Tile size: { 1088, 2, 1 }
// Contraction output var shape: fp32(1, 8, 8, 2080):(133120, 16640, 2080, 1):520 KiB
// Computed true ops: 139264
// Computed work groups: 32
// Computed inner loops: 1
// Computed shared mem: 8712
// Computed out regs: 9216
// Computed mem read: 8704
// Computed mem write: 8704
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1024, 8, 1
__kernel void kernel_c51_sdk_653(__global float* restrict  X_T1999, __global const float* restrict  in1)
{
  X_T1999 = (X_T1999 + 992);
  int tid = get_local_id(0);
  float agg[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[2178];
  int n2_gid = get_group_id(1);
  int n1_gid = (get_group_id(0) * 2);
  {
    {
      int gbase = ((n2_gid * 1088) + (n1_gid * 8704));
      int a_n2_tid = (tid % 256);
      for (int a_n2_lid = 0; a_n2_lid < 5; a_n2_lid += 1)
      {
        int a_n2_cond = ((a_n2_lid < 4) || (a_n2_tid < 64));
        if (a_n2_cond)
        {
          int a_n2 = ((256 * a_n2_lid) + a_n2_tid);
          for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
          {
            int lidx = (a_n2 + (1089 * n1_lid));
            int gidx = ((gbase + a_n2) + (8704 * n1_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)69631)];
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 128);
    int n1_tid = ((tid / 128) % 2);
    for (int a_lid = 0; a_lid < 9; a_lid += 1)
    {
      int a_cond = ((a_lid < 8) || (a_tid < 64));
      int a = select((int)0, (int)((128 * a_lid) + a_tid), (int)a_cond);
      float val1 = in1_shared[(a + (1089 * n1_tid))];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)a_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 128);
  int n1_tid = ((tid / 128) % 2);
  for (int a_lid = 0; a_lid < 9; a_lid += 1)
  {
    int a_cond = ((a_lid < 8) || (a_tid < 64));
    if (a_cond)
    {
      int a = ((128 * a_lid) + a_tid);
      float LX_T1999 = agg[a_lid];
      int gout_idx = ((a + (16640 * (n1_gid + n1_tid))) + (2080 * n2_gid));
      X_T1999[gout_idx] = LX_T1999;
    }
  }
}
