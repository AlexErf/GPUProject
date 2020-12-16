#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 7 1
// lid: 256 1 1
// Original:
// X_T1493[n0, n1, n2, a : _T2123, _T2124, _T2125, _T2126] = =(X_T1492[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T1493[n0, n1, n2, a : _T2123, _T2124, _T2125, _T2126] = =(X_T1492[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 768, 0 <= a < 800, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 7, 0 <= n2 < 7, 0 <= a < 768 }
// Defracted:
// X_T1493[n0, n1, n2, a : _T2123, _T2124, _T2125, _T2126] = =(X_T1492[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T1493   X_T1492  
//        a       768         1         1  
//       n1         7      5600      5376  
//       n2         7       800       768  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 768, 7, 7 }
// Out stride: { 1, 5600, 800 }
// Input 1 offset: 0
// Input 1 stride: { 1, 5376, 768 }
// Tile size: { 128, 1, 7 }
// Contraction output var shape: fp32(1, 7, 7, 800):(39200, 5600, 800, 1):153.125 KiB
// Computed true ops: 75264
// Computed work groups: 42
// Computed inner loops: 1
// Computed shared mem: 3584
// Computed out regs: 4096
// Computed mem read: 3584
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 7, 1
__kernel void kernel_c108_sdk_510(__global float* restrict  X_T1493, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[896];
  int a_gid = (get_group_id(0) * 128);
  int n1_gid = get_group_id(1);
  {
    {
      int gbase = (a_gid + (n1_gid * 5376));
      int a_tid = (tid % 128);
      int n2_n1_tid = ((tid / 128) % 2);
      for (int n2_n1_lid = 0; n2_n1_lid < 4; n2_n1_lid += 1)
      {
        int n2_n1_cond = ((n2_n1_lid < 3) || (n2_n1_tid < 1));
        if (n2_n1_cond)
        {
          int n2_n1 = ((2 * n2_n1_lid) + n2_n1_tid);
          int lidx = ((7 * a_tid) + n2_n1);
          int gidx = ((gbase + a_tid) + (768 * n2_n1));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)37631)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 8);
    int n2_cond = (n2_tid < 7);
    int n2 = select((int)0, (int)n2_tid, (int)n2_cond);
    for (int a_lid = 0; a_lid < 4; a_lid += 1)
    {
      int a = ((32 * a_lid) + a_tid);
      float val1 = in1_shared[((7 * a) + n2)];
      agg[a_lid] = select((float)agg[a_lid], (float)val1, (int)n2_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 8);
  int n2_cond = (n2_tid < 7);
  if (n2_cond)
  {
    for (int a_lid = 0; a_lid < 4; a_lid += 1)
    {
      int a = ((32 * a_lid) + a_tid);
      float LX_T1493 = agg[a_lid];
      int gout_idx = (((a_gid + a) + (5600 * n1_gid)) + (800 * n2_tid));
      X_T1493[gout_idx] = LX_T1493;
    }
  }
}
