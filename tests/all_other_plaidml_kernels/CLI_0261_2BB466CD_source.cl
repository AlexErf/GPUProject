#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T2342[n, x0, x1, c : _T3706, _T3707, _T3708, _T3709] = +(X_T2128[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1
// With Index Variables Made Integral:
// X_T2342[n, x0, x1, c : _T3706, _T3707, _T3708, _T3709] = +(X_T2128[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= k0 < 1, 0 <= k1 < 1, 0 <= n < 1, 0 <= n < 1, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= k0 + 2*x0 < 14, 0 <= k1 + 2*x1 < 14, 0 <= c < 528, 0 <= c < 528, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= k0 < 1, 0 <= k1 < 1, 0 <= n < 1, 0 <= x0 < 7, 0 <= x1 < 7, 0 <= k0 + 2*x0 < 14, 0 <= k1 + 2*x1 < 14, 0 <= c < 528 }
// Defracted:
// X_T2342[n, x0, x1, c : _T3706, _T3707, _T3708, _T3709] = +(X_T2128[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T2342   X_T2128  
//        c       528         1         1  
//       x0         7      3696     14784  
//       x1         7       528      1056  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, x0, x1 }
// Ranges: { 528, 7, 7 }
// Out stride: { 1, 3696, 528 }
// Input 1 offset: 0
// Input 1 stride: { 1, 14784, 1056 }
// Tile size: { 528, 1, 1 }
// Contraction output var shape: fp32(1, 7, 7, 528):(25872, 3696, 528, 1):101.062 KiB
// Computed true ops: 51744
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 2112
// Computed out regs: 3072
// Computed mem read: 2048
// Computed mem write: 2176
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c42_sdk_896(__global float* restrict  X_T2342, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[3] = {0, 0, 0, };
  __local float in1_shared[528];
  int x1_gid = get_group_id(0);
  int x0_gid = get_group_id(1);
  {
    {
      int gbase = ((x1_gid * 1056) + (x0_gid * 14784));
      int c_tid = (tid % 256);
      for (int c_lid = 0; c_lid < 3; c_lid += 1)
      {
        int c_cond = ((c_lid < 2) || (c_tid < 16));
        if (c_cond)
        {
          int c = ((256 * c_lid) + c_tid);
          int gidx = (gbase + c);
          in1_shared[c] = in1[clamp((int)gidx, (int)0, (int)103487)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int c_tid = (tid % 256);
    for (int c_lid = 0; c_lid < 3; c_lid += 1)
    {
      int c_cond = ((c_lid < 2) || (c_tid < 16));
      int c = select((int)0, (int)((256 * c_lid) + c_tid), (int)c_cond);
      float val1 = in1_shared[c];
      float agg_rhs = (agg[c_lid] + val1);
      agg[c_lid] = select((float)agg[c_lid], (float)agg_rhs, (int)c_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 256);
  for (int c_lid = 0; c_lid < 3; c_lid += 1)
  {
    int c_cond = ((c_lid < 2) || (c_tid < 16));
    if (c_cond)
    {
      int c = ((256 * c_lid) + c_tid);
      float LX_T2342 = agg[c_lid];
      int gout_idx = ((c + (3696 * x0_gid)) + (528 * x1_gid));
      X_T2342[gout_idx] = LX_T2342;
    }
  }
}
