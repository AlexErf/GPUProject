#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 7 1
// lid: 256 1 1
// Original:
// X_T318[n, x0, x1, c : _T477, _T478, _T479, _T480] = +(X_T317[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3
// With Index Variables Made Integral:
// X_T318[n, x0, x1, c : _T477, _T478, _T479, _T480] = +(X_T317[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= c < 22, 0 <= c < 22, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + 2*x0 < 57, 0 <= k1 + 2*x1 < 57, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + k0 < 1000000000, 0 <= 500000000 + k1 < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= k0 < 3, 0 <= k1 < 3, 0 <= c < 22, 0 <= x0 < 28, 0 <= x1 < 28, 0 <= k0 + 2*x0 < 57, 0 <= k1 + 2*x1 < 57 }
// Defracted:
// X_T318[n, x0, x1, c : _T477, _T478, _T479, _T480] = +(X_T317[n, k0 + 2*x0, k1 + 2*x1, c]), k0 < 3, k1 < 3, 500000000 + c < 1000000000, 500000000 + k0 < 1000000000, 500000000 + k1 < 1000000000, 500000000 + n < 1000000000, 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000
// Flattened:
//              Range    X_T318    X_T317  
//        c        22         1         1  
//       k0         3         0      1254  
//       k1         3         0        22  
//       x0        28       616      2508  
//       x1        28        22        44  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, k0, k1, x0, x1 }
// Ranges: { 22, 3, 3, 28, 28 }
// Out stride: { 1, 0, 0, 616, 22 }
// Input 1 offset: 0
// Input 1 stride: { 1, 1254, 22, 2508, 44 }
// Elementwise input X_T316 shape: fp32(1, 28, 28, 22):(17248, 616, 22, 1):67.375 KiB
// Elementwise op: [[pid(reduction_A_block_stem_2)]] X_T319 = div(X_T316, X_T318)
// Tile size: { 22, 3, 3, 4, 4 }
// Contraction output var shape: fp32(1, 28, 28, 22):(17248, 616, 22, 1):67.375 KiB
// Computed true ops: 465696
// Computed work groups: 49
// Computed inner loops: 1
// Computed shared mem: 7128
// Computed out regs: 2048
// Computed mem read: 7104
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 7, 1
__kernel void kernel_c42_sdk_108(__global float* restrict  X_T319, __global const float* restrict  in1, __global const float* restrict  X_T316)
{
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[1782];
  int x1_gid = (get_group_id(0) * 4);
  int x0_gid = (get_group_id(1) * 4);
  for (int k0_gid = 0; k0_gid < 3; k0_gid += 3)
  {
    for (int k1_gid = 0; k1_gid < 3; k1_gid += 3)
    {
      {
        int gbase = ((((k1_gid * 22) + (x1_gid * 44)) + (k0_gid * 1254)) + (x0_gid * 2508));
        int c_k1_x1_tid = (tid % 256);
        int c_k1_x1_cond = (c_k1_x1_tid < 198);
        if (c_k1_x1_cond)
        {
          for (int k0_x0_lid = 0; k0_x0_lid < 9; k0_x0_lid += 1)
          {
            int lidx = ((9 * c_k1_x1_tid) + k0_x0_lid);
            int gidx = ((gbase + c_k1_x1_tid) + (1254 * k0_x0_lid));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)71477)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int k0_lid = 0; k0_lid < 3; k0_lid += 1)
      {
        for (int k1_lid = 0; k1_lid < 3; k1_lid += 1)
        {
          int c_tid = (tid % 32);
          int x1_tid = ((tid / 32) % 4);
          int x0_tid = ((tid / 128) % 2);
          int c_cond = (c_tid < 22);
          int c = select((int)0, (int)c_tid, (int)c_cond);
          for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
          {
            int x0 = ((2 * x0_lid) + x0_tid);
            float val1 = in1_shared[(((((9 * c) + (198 * k1_lid)) + (396 * x1_tid)) + k0_lid) + (2 * x0))];
            float agg_rhs = (agg[x0_lid] + val1);
            agg[x0_lid] = select((float)agg[x0_lid], (float)agg_rhs, (int)c_cond);
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  int c_tid = (tid % 32);
  int x1_tid = ((tid / 32) % 4);
  int x0_tid = ((tid / 128) % 2);
  int c_cond = (c_tid < 22);
  if (c_cond)
  {
    for (int x0_lid = 0; x0_lid < 2; x0_lid += 1)
    {
      int x0 = ((2 * x0_lid) + x0_tid);
      float LX_T318 = agg[x0_lid];
      int gout_idx = ((c_tid + (616 * (x0_gid + x0))) + (22 * (x1_gid + x1_tid)));
      float LX_T316 = X_T316[gout_idx];
      float LX_T319 = (LX_T316 / LX_T318);
      X_T319[gout_idx] = LX_T319;
    }
  }
}
