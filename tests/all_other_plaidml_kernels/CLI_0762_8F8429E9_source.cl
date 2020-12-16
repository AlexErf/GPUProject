#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7680 1 1
// lid: 256 1 1
// Original:
// X_T2611[x0, x3 : _T3761, _T3762] = +(X_T2610[x0, x1, x2, x3])
// With Index Variables Made Integral:
// X_T2611[x0, x3 : _T3761, _T3762] = +(X_T2610[x0, x1, x2, x3]), 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000, 500000000 + x2 < 1000000000, 500000000 + x3 < 1000000000
// Constraints:{ 0 <= x0 < 1, 0 <= x0 < 1, 0 <= x1 < 7, 0 <= x2 < 7, 0 <= x3 < 1920, 0 <= x3 < 1920, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000, 0 <= 500000000 + x2 < 1000000000, 0 <= 500000000 + x3 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= x0 < 1, 0 <= x1 < 7, 0 <= x2 < 7, 0 <= x3 < 1920 }
// Defracted:
// X_T2611[x0, x3 : _T3761, _T3762] = +(X_T2610[x0, x1, x2, x3]), 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000, 500000000 + x2 < 1000000000, 500000000 + x3 < 1000000000
// Flattened:
//              Range   X_T2611   X_T2610  
//       x1         7         0     13440  
//       x2         7         0      1920  
//       x3      1920         1         1  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { x1, x2, x3 }
// Ranges: { 7, 7, 1920 }
// Out stride: { 0, 0, 1 }
// Input 1 offset: 0
// Input 1 stride: { 13440, 1920, 1 }
// Elementwise op: [[pid(Mean)]] X_T2613 = div(X_T2611, X_T2612)
// Tile size: { 7, 7, 64 }
// Contraction output var shape: fp32(1, 1920):(1920, 1):7.5 KiB
// Computed true ops: 282240
// Computed work groups: 30
// Computed inner loops: 1
// Computed shared mem: 13568
// Computed out regs: 1024
// Computed mem read: 12544
// Computed mem write: 256
// Computed operations: 256
// Computed rollups: 2
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7680, 1, 1
__kernel void kernel_c124_sdk_908(__global float* restrict  X_T2613, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[3136];
  int x3_gid = (get_group_id(0) * 64);
  for (int x1_gid = 0; x1_gid < 7; x1_gid += 7)
  {
    for (int x2_gid = 0; x2_gid < 7; x2_gid += 7)
    {
      {
        int gbase = ((x3_gid + (x2_gid * 1920)) + (x1_gid * 13440));
        int x3_tid = (tid % 64);
        int x2_x1_tid = ((tid / 64) % 4);
        for (int x2_x1_lid = 0; x2_x1_lid < 13; x2_x1_lid += 1)
        {
          int x2_x1_cond = ((x2_x1_lid < 12) || (x2_x1_tid < 1));
          if (x2_x1_cond)
          {
            int x2_x1 = ((4 * x2_x1_lid) + x2_x1_tid);
            int lidx = ((49 * x3_tid) + x2_x1);
            int gidx = ((gbase + x3_tid) + (1920 * x2_x1));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)94079)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      int x1_tid = ((tid / 64) % 4);
      for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
      {
        int x1_cond = ((x1_lid < 1) || (x1_tid < 3));
        if (x1_cond)
        {
          int x1 = ((4 * x1_lid) + x1_tid);
          for (int x2_lid = 0; x2_lid < 7; x2_lid += 1)
          {
            int x3_tid = (tid % 64);
            float val1 = in1_shared[(((49 * x3_tid) + x2_lid) + (7 * x1))];
            float agg_rhs = (agg[0] + val1);
            agg[0] = agg_rhs;
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
  __local float merge_shared[256];
  {
    merge_shared[tid] = agg[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 128))
    {
      merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 128)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 64))
    {
      merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 64)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 64))
    {
      agg[0] = merge_shared[tid];
    }
  }
  int x3_tid = (tid % 64);
  if ((tid < 64))
  {
    float LX_T2611 = agg[0];
    int gout_idx = (x3_gid + x3_tid);
    float LX_T2613 = (LX_T2611 / (float)49);
    X_T2613[gout_idx] = LX_T2613;
  }
}
