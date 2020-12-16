#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 6144 1 1
// lid: 256 1 1
// Original:
// X_T2503[x0, x3 : _T3519, _T3520] = +(X_T2502[x0, x1, x2, x3])
// With Index Variables Made Integral:
// X_T2503[x0, x3 : _T3519, _T3520] = +(X_T2502[x0, x1, x2, x3]), 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000, 500000000 + x2 < 1000000000, 500000000 + x3 < 1000000000
// Constraints:{ 0 <= x0 < 1, 0 <= x0 < 1, 0 <= x1 < 8, 0 <= x2 < 8, 0 <= x3 < 1536, 0 <= x3 < 1536, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000, 0 <= 500000000 + x2 < 1000000000, 0 <= 500000000 + x3 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= x0 < 1, 0 <= x1 < 8, 0 <= x2 < 8, 0 <= x3 < 1536 }
// Defracted:
// X_T2503[x0, x3 : _T3519, _T3520] = +(X_T2502[x0, x1, x2, x3]), 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000, 500000000 + x2 < 1000000000, 500000000 + x3 < 1000000000
// Flattened:
//              Range   X_T2503   X_T2502  
//       x1         8         0     12288  
//       x2         8         0      1536  
//       x3      1536         1         1  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { x1, x2, x3 }
// Ranges: { 8, 8, 1536 }
// Out stride: { 0, 0, 1 }
// Input 1 offset: 0
// Input 1 stride: { 12288, 1536, 1 }
// Elementwise op: [[pid(Mean)]] X_T2505 = div(X_T2503, X_T2504)
// Tile size: { 1, 8, 64 }
// Contraction output var shape: fp32(1, 1536):(1536, 1):6 KiB
// Computed true ops: 294912
// Computed work groups: 24
// Computed inner loops: 8
// Computed shared mem: 3104
// Computed out regs: 1024
// Computed mem read: 2048
// Computed mem write: 256
// Computed operations: 256
// Computed rollups: 2
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 6144, 1, 1
__kernel void kernel_c51_sdk_818(__global float* restrict  X_T2505, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[520];
  int x3_gid = (get_group_id(0) * 64);
  for (int x1_gid = 0; x1_gid < 8; x1_gid += 1)
  {
    for (int x2_gid = 0; x2_gid < 8; x2_gid += 8)
    {
      {
        int gbase = ((x3_gid + (x2_gid * 1536)) + (x1_gid * 12288));
        int x3_tid = (tid % 64);
        int x2_x1_tid = ((tid / 64) % 4);
        for (int x2_x1_lid = 0; x2_x1_lid < 2; x2_x1_lid += 1)
        {
          int x2_x1 = ((4 * x2_x1_lid) + x2_x1_tid);
          int lidx = (x3_tid + (65 * x2_x1));
          int gidx = ((gbase + x3_tid) + (1536 * x2_x1));
          in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)98303)];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      int x2_tid = ((tid / 64) % 4);
      for (int x2_lid = 0; x2_lid < 2; x2_lid += 1)
      {
        int x2 = ((4 * x2_lid) + x2_tid);
        int x3_tid = (tid % 64);
        float val1 = in1_shared[(x3_tid + (65 * x2))];
        float agg_rhs = (agg[0] + val1);
        agg[0] = agg_rhs;
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
    float LX_T2503 = agg[0];
    int gout_idx = (x3_gid + x3_tid);
    float LX_T2505 = (LX_T2503 / (float)64);
    X_T2505[gout_idx] = LX_T2505;
  }
}
