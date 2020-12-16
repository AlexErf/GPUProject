#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 19712 1 1
// lid: 256 1 1
// Original:
// X_T1211[n, d0, d1, c : _T1895, _T1896, _T1897, _T1898] = =(X_T1209[n, d0, d1, c])
// With Index Variables Made Integral:
// X_T1211[n, d0, d1, c : _T1895, _T1896, _T1897, _T1898] = =(X_T1209[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= d0 < 28, 0 <= d1 < 28, 0 <= d0 < 29, 0 <= d1 < 29, 0 <= c < 88, 0 <= c < 88, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= d0 < 28, 0 <= d1 < 28, 0 <= c < 88 }
// Defracted:
// X_T1211[n, d0, d1, c : _T1895, _T1896, _T1897, _T1898] = =(X_T1209[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range   X_T1211   X_T1209  
//        c        88         1         1  
//       d0        28      2552      2464  
//       d1        28        88        88  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 2464, 28 }
// Out stride: { 1, 2552 }
// Input 1 offset: 0
// Input 1 stride: { 1, 2464 }
// Tile size: { 32, 28 }
// Contraction output var shape: fp32(1, 29, 29, 88):(74008, 2552, 88, 1):289.094 KiB
// Computed true ops: 137984
// Computed work groups: 77
// Computed inner loops: 1
// Computed shared mem: 3696
// Computed out regs: 4096
// Computed mem read: 3584
// Computed mem write: 3584
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 19712, 1, 1
__kernel void kernel_c42_sdk_455(__global float* restrict  X_T1211, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[924];
  int d1_c_gid = (get_group_id(0) * 32);
  {
    {
      int d1_c_tid = (tid % 32);
      int d0_tid = ((tid / 32) % 8);
      for (int d0_lid = 0; d0_lid < 4; d0_lid += 1)
      {
        int d0_cond = ((d0_lid < 3) || (d0_tid < 4));
        if (d0_cond)
        {
          int d0 = ((8 * d0_lid) + d0_tid);
          int lidx = (d1_c_tid + (33 * d0));
          int gidx = ((d1_c_gid + d1_c_tid) + (2464 * d0));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)68991)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int d1_c_tid = (tid % 32);
    int d0_tid = ((tid / 32) % 8);
    for (int d0_lid = 0; d0_lid < 4; d0_lid += 1)
    {
      int d0_cond = ((d0_lid < 3) || (d0_tid < 4));
      int d0 = select((int)0, (int)((8 * d0_lid) + d0_tid), (int)d0_cond);
      float val1 = in1_shared[(d1_c_tid + (33 * d0))];
      agg[d0_lid] = select((float)agg[d0_lid], (float)val1, (int)d0_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int d1_c_tid = (tid % 32);
  int d0_tid = ((tid / 32) % 8);
  for (int d0_lid = 0; d0_lid < 4; d0_lid += 1)
  {
    int d0_cond = ((d0_lid < 3) || (d0_tid < 4));
    if (d0_cond)
    {
      int d0 = ((8 * d0_lid) + d0_tid);
      float LX_T1211 = agg[d0_lid];
      int gout_idx = ((d1_c_gid + d1_c_tid) + (2552 * d0));
      X_T1211[gout_idx] = LX_T1211;
    }
  }
}
