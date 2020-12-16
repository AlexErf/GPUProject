#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 2816 11 1
// lid: 256 1 1
// Original:
// X_T3029[n, x0, x1, c : _T4818, _T4819, _T4820, _T4821] = +(X_T2660[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1
// With Index Variables Made Integral:
// X_T3029[n, x0, x1, c : _T4818, _T4819, _T4820, _T4821] = +(X_T2660[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= k0 < 1, 0 <= k1 < 1, 0 <= n < 1, 0 <= n < 1, 0 <= x0 < 11, 0 <= x1 < 11, 0 <= k0 + 2*x0 < 21, 0 <= k1 + 2*x1 < 21, 0 <= c < 2016, 0 <= c < 2016, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= k0 < 1, 0 <= k1 < 1, 0 <= n < 1, 0 <= x0 < 11, 0 <= x1 < 11, 0 <= k0 + 2*x0 < 21, 0 <= k1 + 2*x1 < 21, 0 <= c < 2016 }
// Defracted:
// X_T3029[n, x0, x1, c : _T4818, _T4819, _T4820, _T4821] = +(X_T2660[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 1, k1 < 1, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range   X_T3029   X_T2660  
//        c      2016         1         1  
//       x0        11     22176     84672  
//       x1        11      2016      4032  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, x0, x1 }
// Ranges: { 2016, 11, 11 }
// Out stride: { 1, 22176, 2016 }
// Input 1 offset: 0
// Input 1 stride: { 1, 84672, 4032 }
// Tile size: { 2016, 1, 1 }
// Contraction output var shape: fp32(1, 11, 11, 2016):(243936, 22176, 2016, 1):952.875 KiB
// Computed true ops: 487872
// Computed work groups: 121
// Computed inner loops: 1
// Computed shared mem: 8064
// Computed out regs: 8192
// Computed mem read: 8064
// Computed mem write: 8064
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 2816, 11, 1
__kernel void kernel_c42_sdk_1172(__global float* restrict  X_T3029, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[8] = {0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[2016];
  int x1_gid = get_group_id(0);
  int x0_gid = get_group_id(1);
  {
    {
      int gbase = ((x1_gid * 4032) + (x0_gid * 84672));
      int c_tid = (tid % 256);
      for (int c_lid = 0; c_lid < 8; c_lid += 1)
      {
        int c_cond = ((c_lid < 7) || (c_tid < 224));
        if (c_cond)
        {
          int c = ((256 * c_lid) + c_tid);
          int gidx = (gbase + c);
          in1_shared[c] = in1[clamp((int)gidx, (int)0, (int)889055)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int c_tid = (tid % 256);
    for (int c_lid = 0; c_lid < 8; c_lid += 1)
    {
      int c_cond = ((c_lid < 7) || (c_tid < 224));
      int c = select((int)0, (int)((256 * c_lid) + c_tid), (int)c_cond);
      float val1 = in1_shared[c];
      float agg_rhs = (agg[c_lid] + val1);
      agg[c_lid] = select((float)agg[c_lid], (float)agg_rhs, (int)c_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 256);
  for (int c_lid = 0; c_lid < 8; c_lid += 1)
  {
    int c_cond = ((c_lid < 7) || (c_tid < 224));
    if (c_cond)
    {
      int c = ((256 * c_lid) + c_tid);
      float LX_T3029 = agg[c_lid];
      int gout_idx = ((c + (22176 * x0_gid)) + (2016 * x1_gid));
      X_T3029[gout_idx] = LX_T3029;
    }
  }
}
