#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 21 1
// lid: 256 1 1
// Original:
// X_T878[n0, n1, n2, a : _T1241, _T1242, _T1243, _T1244] = =(X_T877[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T878[n0, n1, n2, a : _T1241, _T1242, _T1243, _T1244] = =(X_T877[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 672, 0 <= a < 704, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 672 }
// Defracted:
// X_T878[n0, n1, n2, a : _T1241, _T1242, _T1243, _T1244] = =(X_T877[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T878    X_T877  
//        a       672         1         1  
//       n1        14      9856      9408  
//       n2        14       704       672  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 672, 14, 14 }
// Out stride: { 1, 9856, 704 }
// Input 1 offset: 0
// Input 1 stride: { 1, 9408, 672 }
// Tile size: { 32, 2, 14 }
// Contraction output var shape: fp32(1, 14, 14, 704):(137984, 9856, 704, 1):539 KiB
// Computed true ops: 263424
// Computed work groups: 147
// Computed inner loops: 1
// Computed shared mem: 3696
// Computed out regs: 4096
// Computed mem read: 3584
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 21, 1
__kernel void kernel_c68_sdk_297(__global float* restrict  X_T878, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[924];
  int a_gid = (get_group_id(1) * 32);
  int n1_gid = (get_group_id(0) * 2);
  {
    {
      int gbase = (a_gid + (n1_gid * 9408));
      int a_tid = (tid % 32);
      int n2_n1_tid = ((tid / 32) % 8);
      for (int n2_n1_lid = 0; n2_n1_lid < 4; n2_n1_lid += 1)
      {
        int n2_n1_cond = ((n2_n1_lid < 3) || (n2_n1_tid < 4));
        if (n2_n1_cond)
        {
          int n2_n1 = ((8 * n2_n1_lid) + n2_n1_tid);
          int lidx = (a_tid + (33 * n2_n1));
          int gidx = ((gbase + a_tid) + (672 * n2_n1));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)131711)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 4);
    int n1_tid = ((tid / 128) % 2);
    for (int n2_lid = 0; n2_lid < 4; n2_lid += 1)
    {
      int n2_cond = ((n2_lid < 3) || (n2_tid < 2));
      int n2 = select((int)0, (int)((4 * n2_lid) + n2_tid), (int)n2_cond);
      float val1 = in1_shared[((a_tid + (33 * n2)) + (462 * n1_tid))];
      agg[n2_lid] = select((float)agg[n2_lid], (float)val1, (int)n2_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 4);
  int n1_tid = ((tid / 128) % 2);
  for (int n2_lid = 0; n2_lid < 4; n2_lid += 1)
  {
    int n2_cond = ((n2_lid < 3) || (n2_tid < 2));
    if (n2_cond)
    {
      int n2 = ((4 * n2_lid) + n2_tid);
      float LX_T878 = agg[n2_lid];
      int gout_idx = (((a_gid + a_tid) + (9856 * (n1_gid + n1_tid))) + (704 * n2));
      X_T878[gout_idx] = LX_T878;
    }
  }
}
