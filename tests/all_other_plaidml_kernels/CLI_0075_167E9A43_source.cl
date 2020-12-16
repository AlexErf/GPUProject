#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 8192 1 1
// lid: 256 1 1
// Original:
// X_T681[x0, x3 : _T820, _T821] = +(X_T680[x0, x1, x2, x3])
// With Index Variables Made Integral:
// X_T681[x0, x3 : _T820, _T821] = +(X_T680[x0, x1, x2, x3]), 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000, 500000000 + x2 < 1000000000, 500000000 + x3 < 1000000000
// Constraints:{ 0 <= x0 < 1, 0 <= x0 < 1, 0 <= x1 < 7, 0 <= x2 < 7, 0 <= x3 < 2048, 0 <= x3 < 2048, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000, 0 <= 500000000 + x2 < 1000000000, 0 <= 500000000 + x3 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= x0 < 1, 0 <= x1 < 7, 0 <= x2 < 7, 0 <= x3 < 2048 }
// Defracted:
// X_T681[x0, x3 : _T820, _T821] = +(X_T680[x0, x1, x2, x3]), 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000, 500000000 + x2 < 1000000000, 500000000 + x3 < 1000000000
// Flattened:
//              Range    X_T681    X_T680  
//       x1         7         0     14336  
//       x2         7         0      2048  
//       x3      2048         1         1  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { x1, x2, x3 }
// Ranges: { 7, 7, 2048 }
// Out stride: { 0, 0, 1 }
// Input 1 offset: 0
// Input 1 stride: { 14336, 2048, 1 }
// Elementwise op: [[pid(Mean)]] X_T683 = div(X_T681, X_T682)
// Tile size: { 7, 7, 64 }
// Contraction output var shape: fp32(1, 2048):(2048, 1):8 KiB
// Computed true ops: 301056
// Computed work groups: 32
// Computed inner loops: 1
// Computed shared mem: 13568
// Computed out regs: 1024
// Computed mem read: 12544
// Computed mem write: 256
// Computed operations: 256
// Computed rollups: 2
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 8192, 1, 1
__kernel void kernel_c29_sdk_162(__global float* restrict  X_T683, __global const float* restrict  in1)
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
        int gbase = ((x3_gid + (x2_gid * 2048)) + (x1_gid * 14336));
        int x3_tid = (tid % 64);
        int x2_x1_tid = ((tid / 64) % 4);
        for (int x2_x1_lid = 0; x2_x1_lid < 13; x2_x1_lid += 1)
        {
          int x2_x1_cond = ((x2_x1_lid < 12) || (x2_x1_tid < 1));
          if (x2_x1_cond)
          {
            int x2_x1 = ((4 * x2_x1_lid) + x2_x1_tid);
            int lidx = ((49 * x3_tid) + x2_x1);
            int gidx = ((gbase + x3_tid) + (2048 * x2_x1));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)100351)];
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
    float LX_T681 = agg[0];
    int gout_idx = (x3_gid + x3_tid);
    float LX_T683 = (LX_T681 / (float)49);
    X_T683[gout_idx] = LX_T683;
  }
}
