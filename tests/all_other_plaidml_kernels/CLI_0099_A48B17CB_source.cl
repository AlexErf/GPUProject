#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 16384 1 1
// lid: 256 1 1
// Original:
// X_T633[x0, x3 : _T838, _T839] = +(X_T632[x0, x1, x2, x3])
// With Index Variables Made Integral:
// X_T633[x0, x3 : _T838, _T839] = +(X_T632[x0, x1, x2, x3]), 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000, 500000000 + x2 < 1000000000, 500000000 + x3 < 1000000000
// Constraints:{ 0 <= x0 < 1, 0 <= x0 < 1, 0 <= x1 < 10, 0 <= x2 < 10, 0 <= x3 < 2048, 0 <= x3 < 2048, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000, 0 <= 500000000 + x2 < 1000000000, 0 <= 500000000 + x3 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= x0 < 1, 0 <= x1 < 10, 0 <= x2 < 10, 0 <= x3 < 2048 }
// Defracted:
// X_T633[x0, x3 : _T838, _T839] = +(X_T632[x0, x1, x2, x3]), 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000, 500000000 + x2 < 1000000000, 500000000 + x3 < 1000000000
// Flattened:
//              Range    X_T633    X_T632  
//       x1        10         0     20480  
//       x2        10         0      2048  
//       x3      2048         1         1  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { x1, x2, x3 }
// Ranges: { 10, 10, 2048 }
// Out stride: { 0, 0, 1 }
// Input 1 offset: 0
// Input 1 stride: { 20480, 2048, 1 }
// Elementwise op: [[pid(Mean)]] X_T635 = div(X_T633, X_T634)
// Tile size: { 10, 10, 32 }
// Contraction output var shape: fp32(1, 2048):(2048, 1):8 KiB
// Computed true ops: 614400
// Computed work groups: 64
// Computed inner loops: 1
// Computed shared mem: 13952
// Computed out regs: 1024
// Computed mem read: 12800
// Computed mem write: 128
// Computed operations: 256
// Computed rollups: 3
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 16384, 1, 1
__kernel void kernel_c28_sdk_192(__global float* restrict  X_T635, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[3232];
  int x3_gid = (get_group_id(0) * 32);
  for (int x1_gid = 0; x1_gid < 10; x1_gid += 10)
  {
    for (int x2_gid = 0; x2_gid < 10; x2_gid += 10)
    {
      {
        int gbase = ((x3_gid + (x2_gid * 2048)) + (x1_gid * 20480));
        int x3_tid = (tid % 32);
        int x2_x1_tid = ((tid / 32) % 8);
        for (int x2_x1_lid = 0; x2_x1_lid < 13; x2_x1_lid += 1)
        {
          int x2_x1_cond = ((x2_x1_lid < 12) || (x2_x1_tid < 4));
          if (x2_x1_cond)
          {
            int x2_x1 = ((8 * x2_x1_lid) + x2_x1_tid);
            int lidx = ((101 * x3_tid) + x2_x1);
            int gidx = ((gbase + x3_tid) + (2048 * x2_x1));
            in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)204799)];
          }
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      int x1_tid = ((tid / 32) % 8);
      for (int x1_lid = 0; x1_lid < 2; x1_lid += 1)
      {
        int x1_cond = ((x1_lid < 1) || (x1_tid < 2));
        if (x1_cond)
        {
          int x1 = ((8 * x1_lid) + x1_tid);
          for (int x2_lid = 0; x2_lid < 10; x2_lid += 1)
          {
            int x3_tid = (tid % 32);
            float val1 = in1_shared[(((101 * x3_tid) + x2_lid) + (10 * x1))];
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
    float LX_T633 = agg[0];
    int gout_idx = (x3_gid + x3_tid);
    float LX_T635 = (LX_T633 / (float)100);
    X_T635[gout_idx] = LX_T635;
  }
}
