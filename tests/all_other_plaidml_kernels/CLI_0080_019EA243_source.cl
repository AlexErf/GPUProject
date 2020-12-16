#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T96[n, x0, x1, c : _T97, _T98, _T99, _T100] = +(X_T56[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T96[n, x0, x1, c : _T97, _T98, _T99, _T100] = +(X_T56[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= c < 11, 0 <= c < 11, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + 2*x0 < 113, 0 <= k1 + 2*x1 < 113, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= c < 11, 0 <= x0 < 56, 0 <= x1 < 56, 0 <= k0 + 2*x0 < 113, 0 <= k1 + 2*x1 < 113 }
// Defracted:
// X_T96[n, x0, x1, c : _T97, _T98, _T99, _T100] = +(X_T56[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range     X_T96     X_T56  
//        c        11         1         1  
//       k0         3         0      1243  
//       k1         3         0        11  
//       x0        56       616      2486  
//       x1        56        11        22  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 11, 3, 3, 56, 56 }
// Out stride: { 1, 0, 0, 616, 11 }
// Input 1 offset: 0
// Input 1 stride: { 1, 1243, 11, 2486, 22 }
// Tile size: { 11, 3, 3, 8, 8 }
// Contraction output var shape: fp32(1, 56, 56, 11):(34496, 616, 11, 1):134.75 KiB
// Computed true ops: 620928
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 12716
// Computed out regs: 4096
// Computed mem read: 12672
// Computed mem write: 8192
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c42_sdk_20(__global float* restrict  X_T96, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[3179];
  int x1_gid = (get_group_id(0) * 8);
  int x0_gid = (get_group_id(1) * 8);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((k1_gid * 11) + (x1_gid * 22)) + (k0_gid * 1243)) + (x0_gid * 2486));
        int c_k1_x1_tid = (tid % 256);
        int c_k1_x1_cond = (c_k1_x1_tid < 187);
        if (c_k1_x1_cond)
        {
          for (int k0_x0_lid = 0; k0_x0_lid < 17; k0_x0_lid += 1)
          {
            int lidx = (c_k1_x1_tid + (187 * k0_x0_lid));
            int gidx = ((gbase + c_k1_x1_tid) + (1243 * k0_x0_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)140458)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
      {
        for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
        {
          int c_tid = (tid % 16);
          int x1_tid = ((tid / 16) % 4);
          int x0_tid = ((tid / 64) % 4);
          int c_cond = (c_tid < 11);
          int c = select((int)0, (int)c_tid, (int)c_cond);
          for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
          {
            int x1 = ((4 * x1_lid) + x1_tid);
            for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
            {
              int x0 = ((4 * x0_lid) + x0_tid);
              float val1 = in1_shared[((((c + (11 * k1_lid)) + (22 * x1)) + (187 * k0_lid)) + (374 * x0))];
              int agg_idx = (x1_lid + (x0_lid * 2));
              float agg_rhs = (agg[agg_idx] + val1);
              agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)c_cond);
            }
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 16);
  int x1_tid = ((tid / 16) % 4);
  int x0_tid = ((tid / 64) % 4);
  int c_cond = (c_tid < 11);
  if (c_cond)
  {
    for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
    {
      int x1 = ((4 * x1_lid) + x1_tid);
      for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
      {
        int x0 = ((4 * x0_lid) + x0_tid);
        float LX_T96 = agg[(x1_lid + (x0_lid * 2))];
        int gout_idx = ((c_tid + (616 * (x0_gid + x0))) + (11 * (x1_gid + x1)));
        X_T96[gout_idx] = LX_T96;
      }
    }
  }
}
