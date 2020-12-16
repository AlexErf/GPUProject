#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4096 1 1
// lid: 256 1 1
// Original:
// X_T4094[x0, y1 : _T6538, _T6539] = +(X_T4093[x0, z] * X_I_1[z, y1])
// With Index Variables Made Integral:
// X_T4094[x0, y1 : _T6538, _T6539] = +(X_T4093[x0, z] * X_I_1[z, y1]), 500000000 + x0 < 1000000000, 500000000 + y1 < 1000000000, 500000000 + z < 1000000000
// Constraints:{ 0 <= x0 < 1, 0 <= x0 < 1, 0 <= y1 < 1000, 0 <= y1 < 1000, 0 <= z < 4032, 0 <= z < 4032, 0 <= 500000000 + x0 < 1000000000, 0 <= 500000000 + y1 < 1000000000, 0 <= 500000000 + z < 1000000000 }
// Merged Parallel Constraints:{ 0 <= x0 < 1, 0 <= y1 < 1000, 0 <= z < 4032 }
// Defracted:
// X_T4094[x0, y1 : _T6538, _T6539] = +(X_T4093[x0, z] * X_I_1[z, y1]), 500000000 + x0 < 1000000000, 500000000 + y1 < 1000000000, 500000000 + z < 1000000000
// Flattened:
//              Range   X_T4094   X_T4093     X_I_1  
//       y1      1000         1         0         1  
//        z      4032         0         1      1000  
//      off                   0         0         0  
//      vec                   1         1         1  
// 
// Names: { y1, z }
// Ranges: { 1000, 4032 }
// Out stride: { 1, 0 }
// Input 1 offset: 0
// Input 1 stride: { 0, 1 }
// Input 2 offset: 0
// Input 2 stride: { 1, 1000 }
// Elementwise input X_I_0 shape: fp32(1000):(1):3.90625 KiB
// Elementwise op: [[pid(Add)]] X_T4095 = add(X_T4094, X_I_0)
// Tile size: { 64, 64 }
// Contraction output var shape: fp32(1, 1000):(1000, 1):3.90625 KiB
// Computed true ops: 12096000
// Computed work groups: 16
// Computed inner loops: 63
// Computed shared mem: 17920
// Computed out regs: 1024
// Computed mem read: 16648
// Computed mem write: 256
// Computed operations: 256
// Computed rollups: 2
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4096, 1, 1
__kernel void kernel_c42_sdk_1595(__global float* restrict  X_T4095, __global const float* restrict  in1, __global const float* restrict  in2, __global const float* restrict  X_I_0)
{
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[64];
  __local float in2_shared[4160];
  int y1_gid = (get_group_id(0) * 64);
  for (int z_gid = 0; z_gid < 4032; z_gid += 64)
  {
    {
      int z_tid = (tid % 64);
      if ((tid < 64))
      {
        int gidx = (z_gid + z_tid);
        in1_shared[z_tid] = in1[clamp((int)gidx, (int)0, (int)4031)];
      }
    }
    {
      int gbase = (y1_gid + (z_gid * 1000));
      int y1_tid = (tid % 64);
      int z_tid = ((tid / 64) % 4);
      for (int z_lid = 0; z_lid < 16; z_lid += 1)
      {
        int z = ((4 * z_lid) + z_tid);
        int lidx = (y1_tid + (65 * z));
        int gidx = ((gbase + y1_tid) + (1000 * z));
        in2_shared[lidx] = in2[clamp((int)gidx, (int)0, (int)4031999)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int z_tid = ((tid / 64) % 4);
    for (int z_lid = 0; z_lid < 16; z_lid += 1)
    {
      int z = ((4 * z_lid) + z_tid);
      int y1_tid = (tid % 64);
      float val1 = in1_shared[z];
      float val2 = in2_shared[(y1_tid + (65 * z))];
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
  int y1_tid = (tid % 64);
  int y1_cond = ((y1_gid != 960) || (y1_tid < 40));
  if (y1_cond)
  {
    if ((tid < 64))
    {
      float LX_T4094 = agg[0];
      int gout_idx = (y1_gid + y1_tid);
      float LX_I_0 = X_I_0[gout_idx];
      float LX_T4095 = (LX_T4094 + LX_I_0);
      X_T4095[gout_idx] = LX_T4095;
    }
  }
}
