#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 11 1
// lid: 256 1 1
// Original:
// X_T2879[n, x0, x1, c : _T4573, _T4574, _T4575, _T4576] = +(X_T2832[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T2879[n, x0, x1, c : _T4573, _T4574, _T4575, _T4576] = +(X_T2832[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 11, 0 <= x1 < 11, 0 <= k0 + 2*x0 < 23, 0 <= k1 + 2*x1 < 23, 0 <= c < 672, 0 <= c < 672, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= x0 < 11, 0 <= x1 < 11, 0 <= k0 + 2*x0 < 23, 0 <= k1 + 2*x1 < 23, 0 <= c < 672 }
// Defracted:
// X_T2879[n, x0, x1, c : _T4573, _T4574, _T4575, _T4576] = +(X_T2832[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T2879   X_T2832  
//        c       672         1         1  
//       k0         3         0     15456  
//       k1         3         0       672  
//       x0        11      7392     30912  
//       x1        11       672      1344  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 672, 3, 3, 11, 11 }
// Out stride: { 1, 0, 0, 7392, 672 }
// Input 1 offset: 0
// Input 1 stride: { 1, 15456, 672, 30912, 1344 }
// Tile size: { 256, 3, 1, 11, 1 }
// Contraction output var shape: fp32(1, 11, 11, 672):(81312, 7392, 672, 1):317.625 KiB
// Computed true ops: 1463616
// Computed work groups: 33
// Computed inner loops: 3
// Computed shared mem: 23552
// Computed out regs: 16384
// Computed mem read: 23552
// Computed mem write: 11264
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 11, 1
__kernel void kernel_c42_sdk_1112(__global float* restrict  X_T2879, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[5888];
  int c_gid = (get_group_id(0) * 256);
  int x1_gid = get_group_id(1);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 1)
    {
      {
        int gbase = (((c_gid + (k1_gid * 672)) + (x1_gid * 1344)) + (k0_gid * 15456));
        int c_tid = (tid % 256);
        for (int k0_x0_lid = 0; k0_x0_lid < 23; k0_x0_lid += 1)
        {
          int lidx = ((23 * c_tid) + k0_x0_lid);
          int gidx = ((gbase + c_tid) + (15456 * k0_x0_lid));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)355487)];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
      {
        int c_tid = (tid % 32);
        int x0_tid = ((tid / 32) % 8);
        for (int c_lid = 0; c_lid < 8; c_lid += 1)
        {
          int c = ((32 * c_lid) + c_tid);
          for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
          {
            int x0_cond = ((x0_lid < 1) || (x0_tid < 3));
            int x0 = select((int)0, (int)((8 * x0_lid) + x0_tid), (int)x0_cond);
            float val1 = in1_shared[(((23 * c) + k0_lid) + (2 * x0))];
            int agg_idx = (c_lid + (x0_lid * 8));
            float agg_rhs = (agg[agg_idx] + val1);
            agg[agg_idx] = select((float)agg[agg_idx], (float)agg_rhs, (int)x0_cond);
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 32);
  int x0_tid = ((tid / 32) % 8);
  for (int c_lid = 0; c_lid < 8; c_lid += 1)
  {
    int c_cond = ((c_lid < 5) || (c_gid != 512));
    if (c_cond)
    {
      int c = ((32 * c_lid) + c_tid);
      for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
      {
        int x0_cond = ((x0_lid < 1) || (x0_tid < 3));
        if (x0_cond)
        {
          int x0 = ((8 * x0_lid) + x0_tid);
          float LX_T2879 = agg[(c_lid + (x0_lid * 8))];
          int gout_idx = (((c_gid + c) + (7392 * x0)) + (672 * x1_gid));
          X_T2879[gout_idx] = LX_T2879;
        }
      }
    }
  }
}
