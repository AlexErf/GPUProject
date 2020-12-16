#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Original:
// X_T3135[i, 0 : _T4877, _T4878] = +(X_T3133[i, j])
// With Index Variables Made Integral:
// X_T3135[i, 0 : _T4877, _T4878] = +(X_T3133[i, j]), 500000000 + i < 1000000000, 500000000 + j < 1000000000
// Constraints:{ 0 <= i < 1, 0 <= 0 < 1, 0 <= i < 1, 0 <= j < 1000, 0 <= 500000000 + i < 1000000000, 0 <= 500000000 + j < 1000000000 }
// Merged Parallel Constraints:{ 0 <= i < 1, 0 <= j < 1000 }
// Defracted:
// X_T3135[i, 0 : _T4877, _T4878] = +(X_T3133[i, j]), 500000000 + i < 1000000000, 500000000 + j < 1000000000
// Flattened:
//              Range   X_T3135   X_T3133  
//        j      1000         0         1  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { j }
// Ranges: { 1000 }
// Out stride: { 0 }
// Input 1 offset: 0
// Input 1 stride: { 1 }
// Tile size: { 1000 }
// Contraction output var shape: fp32(1, 1):(1, 1):4 bytes
// Computed true ops: 2000
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 5024
// Computed out regs: 1024
// Computed mem read: 3968
// Computed mem write: 128
// Computed operations: 256
// Computed rollups: 8
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c42_sdk_1184(__global float* restrict  X_T3135, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[1000];
  for (int j_gid = 0; j_gid < 1000; j_gid += 1000)
  {
    {
      int j_tid = (tid % 256);
      for (int j_lid = 0; j_lid < 4; j_lid += 1)
      {
        int j_cond = ((j_lid < 3) || (j_tid < 232));
        if (j_cond)
        {
          int j = ((256 * j_lid) + j_tid);
          int gidx = (j_gid + j);
          in1_shared[j] = in1[clamp((int)gidx, (int)0, (int)999)];
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int j_tid = (tid % 256);
    for (int j_lid = 0; j_lid < 4; j_lid += 1)
    {
      int j_cond = ((j_lid < 3) || (j_tid < 232));
      if (j_cond)
      {
        int j = ((256 * j_lid) + j_tid);
        float val1 = in1_shared[j];
        float agg_rhs = (agg[0] + val1);
        agg[0] = agg_rhs;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
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
    float LX_T3135 = agg[0];
    X_T3135[0] = LX_T3135;
  }
}
