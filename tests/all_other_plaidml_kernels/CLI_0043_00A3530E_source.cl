#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 8192 1 1
// lid: 256 1 1
// Original:
// X_T94[x0, y1 : _T129, _T130] = +(X_T93[x0, z] * X_I_3[z, y1])
// With Index Variables Made Integral:
// X_T94[x0, y1 : _T129, _T130] = +(X_T93[x0, z] * X_I_3[z, y1]), 500000000 + x0 < 1000000000, 500000000 + y1 < 1000000000, 500000000 + z < 1000000000
// Constraints:{ 0 <= x0 < 1, 0 <= x0 < 1, 0 <= y1 < 4096, 0 <= z < 4096, 0 <= z < 4096, 0 <= y1 < 4096, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + y1 < 1000000000, 0 <= 500000000 + z < 1000000000 }
// Merged Parallel Constraints:{ 0 <= x0 < 1, 0 <= y1 < 4096, 0 <= z < 4096 }
// Defracted:
// X_T94[x0, y1 : _T129, _T130] = +(X_T93[x0, z] * X_I_3[z, y1]), 500000000 + x0 < 1000000000, 500000000 + y1 < 1000000000, 500000000 + z < 1000000000
// Flattened:
//              Range     X_T94     X_T93     X_I_3  
//       y1      4096         1         0         1  
//        z      4096         0         1      4096  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { y1, z }
// Ranges: { 4096, 4096 }
// Out stride: { 1, 0 }
// Input 1 offset: 0
// Input 1 stride: { 0, 1 }
// Input 2 offset: 0
// Input 2 stride: { 1, 4096 }
// Elementwise input X_I_2 shape: fp32(4096):(1):16 KiB
// Elementwise op: [[pid(Add)]] X_T95 = add(X_T94, X_I_2)
// Elementwise op: X_T96 = cmp_lt(X_T95, X_T2)
// Elementwise op: [[pid(Relu)]] X_T97 = cond(X_T96, X_T2, X_T95)
// Tile size: { 128, 32 }
// Contraction output var shape: fp32(1, 4096):(4096, 1):16 KiB
// Computed true ops: 83886080
// Computed work groups: 32
// Computed inner loops: 128
// Computed shared mem: 17664
// Computed out regs: 1024
// Computed mem read: 16528
// Computed mem write: 512
// Computed operations: 256
// Computed rollups: 1
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 8192, 1, 1
__kernel void kernel_c18_sdk_19(__global float* restrict  X_T97, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_2)
{
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[32];
  __local float in2_shared[4128];
  int y1_gid = (get_group_id(0) * 128);
  for (int z_gid = 0; z_gid < 4096; z_gid += 32)
  {
    {
      int z_tid = (tid % 32);
      if ((tid < 32))
      {
        int gidx = (z_gid + z_tid);
        in1_shared[z_tid] = in1[clamp((int)gidx, (int)0, (int)4095)];
      }
    }
    {
      int gbase = (y1_gid + (z_gid * 4096));
      int y1_tid = (tid % 128);
      int z_tid = ((tid / 128) % 2);
      for (int z_lid = 0; z_lid < 16; z_lid += 1)
      {
        int z = ((2 * z_lid) + z_tid);
        int lidx = (y1_tid + (129 * z));
        int gidx = ((gbase + y1_tid) + (4096 * z));
        in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)16777215)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int z_tid = ((tid / 128) % 2);
    for (int z_lid = 0; z_lid < 16; z_lid += 1)
    {
      int z = ((2 * z_lid) + z_tid);
      int y1_tid = (tid % 128);
      float val1 = in1_shared[z];
      float val2 = in2_shared[(y1_tid + (129 * z))];
      float agg_rhs = mad(val2, val1, agg[0]);
      agg[0] = agg_rhs;
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
    if ((tid < 128))
    {
      agg[0] = merge_shared[tid];
    }
  }
  int y1_tid = (tid % 128);
  if ((tid < 128))
  {
    float LX_T94 = agg[0];
    int gout_idx = (y1_gid + y1_tid);
    float LX_I_2 = X_I_2[gout_idx];
    float LX_T95 = (LX_T94 + LX_I_2);
    int LX_T96 = (LX_T95 < 0.0f);
    float LX_T97 = select((float)LX_T95, (float)0.0f, (int)LX_T96);
    X_T97[gout_idx] = LX_T97;
  }
}
