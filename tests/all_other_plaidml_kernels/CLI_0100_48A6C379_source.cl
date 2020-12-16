#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Original:
// X_T350[x0 : _T514] = +(X_T349[x0, x1, x2])
// With Index Variables Made Integral:
// X_T350[x0 : _T514] = +(X_T349[x0, x1, x2]), 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000, 500000000 + x2 < 1000000000
// Constraints:{ 0 <= x0 < 1, 0 <= x0 < 1, 0 <= x1 < 80, 0 <= x2 < 128, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + x1 < 1000000000, 0 <= 500000000 + x2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= x0 < 1, 0 <= x1 < 80, 0 <= x2 < 128 }
// Defracted:
// X_T350[x0 : _T514] = +(X_T349[x0, x1, x2]), 500000000 + x0 < 1000000000, 500000000 + x1 < 1000000000, 500000000 + x2 < 1000000000
// Flattened:
//              Range    X_T350    X_T349  
//       x1        80         0       128  
//       x2       128         0         1  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { x1, x2 }
// Ranges: { 80, 128 }
// Out stride: { 0, 0 }
// Input 1 offset: 0
// Input 1 stride: { 128, 1 }
// Tile size: { 4, 128 }
// Contraction output var shape: fp32(1):(1):4 bytes
// Computed true ops: 20480
// Computed work groups: 1
// Computed inner loops: 20
// Computed shared mem: 3072
// Computed out regs: 1024
// Computed mem read: 2048
// Computed mem write: 128
// Computed operations: 256
// Computed rollups: 8
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c6_sdk_175(__global float* restrict  X_T350, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[512];
  for (int x1_gid = 0; x1_gid < 80; x1_gid += 4)
  {
    for (int x2_gid = 0; x2_gid < 128; x2_gid += 128)
    {
      {
        int gbase = (x2_gid + (x1_gid * 128));
        int x2_x1_tid = (tid % 256);
        for (int x2_x1_lid = 0; x2_x1_lid < 2; x2_x1_lid += 1)
        {
          int x2_x1 = ((256 * x2_x1_lid) + x2_x1_tid);
          int gidx = (gbase + x2_x1);
          in1_shared[x2_x1] = in1[clamp((int)gidx, (int)0, (int)10239)];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      int x1_tid = (tid % 4);
      int x2_tid = ((tid / 4) % 64);
      for (int x2_lid = 0; x2_lid < 2; x2_lid += 1)
      {
        int x2 = ((64 * x2_lid) + x2_tid);
        float val1 = in1_shared[(x2 + (128 * x1_tid))];
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
    if ((tid < 32))
    {
      merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 32)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 16))
    {
      merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 16)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 8))
    {
      merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 8)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 4))
    {
      merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 4)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 2))
    {
      merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 2)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 1))
    {
      merge_shared[tid] = (merge_shared[tid] + merge_shared[(tid + 1)]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((tid < 1))
    {
      agg[0] = merge_shared[tid];
    }
  }
  if ((tid < 1))
  {
    float LX_T350 = agg[0];
    X_T350[0] = LX_T350;
  }
}
