#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 10240 1 1
// lid: 256 1 1
// Original:
// X_T718[x0, x3 : _T861, _T862] = +(X_T717[x0, x1, x2, x3])
// With Index Variables Made Integral:
// X_T718[x0, x3 : _T861, _T862] = +(X_T717[x0, x1, x2, x3]), 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000, 500000000 + x2 < 1000000000, 500000000 + x3 < 1000000000
// Constraints:{ 0 <= x0 < 1, 0 <= x0 < 1, 0 <= x1 < 7, 0 <= x2 < 7, 0 <= x3 < 1280, 0 <= x3 < 1280, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000, 0 <= 500000000 + x2 < 1000000000, 0 <= 500000000 + x3 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= x0 < 1, 0 <= x1 < 7, 0 <= x2 < 7, 0 <= x3 < 1280 }
// Defracted:
// X_T718[x0, x3 : _T861, _T862] = +(X_T717[x0, x1, x2, x3]), 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000, 500000000 + x2 < 1000000000, 500000000 + x3 < 1000000000
// Flattened:
//              Range    X_T718    X_T717  
//       x1         7         0      8960  
//       x2         7         0      1280  
//       x3      1280         1         1  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { x1, x2, x3 }
// Ranges: { 7, 7, 1280 }
// Out stride: { 0, 0, 1 }
// Input 1 offset: 0
// Input 1 stride: { 8960, 1280, 1 }
// Elementwise op: [[pid(Mean)]] X_T720 = div(X_T718, X_T719)
// Tile size: { 7, 7, 32 }
// Contraction output var shape: fp32(1, 1280):(1280, 1):5 KiB
// Computed true ops: 188160
// Computed work groups: 40
// Computed inner loops: 1
// Computed shared mem: 7296
// Computed out regs: 1024
// Computed mem read: 6272
// Computed mem write: 128
// Computed operations: 256
// Computed rollups: 3
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 10240, 1, 1
__kernel void kernel_c43_sdk_195(__global float* restrict  X_T720, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[1568];
  int x3_gid = (get_group_id(0) * 32);
  for (int x1_gid = 0; x1_gid < 7; x1_gid += 7)
  {
    for (int x2_gid = 0; x2_gid < 7; x2_gid += 7)
    {
      {
        int gbase = ((x3_gid + (x2_gid * 1280)) + (x1_gid * 8960));
        int x3_tid = (tid % 32);
        int x2_x1_tid = ((tid / 32) % 8);
        for (int x2_x1_lid = 0; x2_x1_lid < 7; x2_x1_lid += 1)
        {
          int x2_x1_cond = ((x2_x1_lid < 6) || (x2_x1_tid < 1));
          if (x2_x1_cond)
          {
            int x2_x1 = ((8 * x2_x1_lid) + x2_x1_tid);
            int lidx = ((49 * x3_tid) + x2_x1);
            int gidx = ((gbase + x3_tid) + (1280 * x2_x1));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)62719)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      int x1_tid = ((tid / 32) % 8);
      int x1_cond = (x1_tid < 7);
      if (x1_cond)
      {
        for (int x2_lid = 0; x2_lid < 7; x2_lid += 1)
        {
          int x3_tid = (tid % 32);
          float val1 = in1_shared[(((49 * x3_tid) + x2_lid) + (7 * x1_tid))];
          float agg_rhs = (agg[0] + val1);
          agg[0] = agg_rhs;
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
    if ((tid < 32))
    {
      merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 32)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 32))
    {
      agg[0] = merge_shared[tid];
    }
  }
  int x3_tid = (tid % 32);
  if ((tid < 32))
  {
    float LX_T718 = agg[0];
    int gout_idx = (x3_gid + x3_tid);
    float LX_T720 = (LX_T718 / (float)49);
    X_T720[gout_idx] = LX_T720;
  }
}
