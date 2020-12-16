#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 5120 1 1
// lid: 256 1 1
// Original:
// X_T349[i0, i1, i2 : _T511, _T512, _T513] = =(X_I_16[0])
// With Index Variables Made Integral:
// X_T349[i0, i1, i2 : _T511, _T512, _T513] = =(X_I_16[0]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000, 500000000 + i2 < 1000000000
// Constraints:{ 0 <= i0 < 1, 0 <= 0 < 1, 0 <= i1 < 80, 0 <= i2 < 128, 0 <= 500000000 + i0 < 1000000000, 0 <= 500000000 + i1 < 1000000000, 0 <= 500000000 + i2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= i0 < 1, 0 <= i1 < 80, 0 <= i2 < 128 }
// Defracted:
// X_T349[i0, i1, i2 : _T511, _T512, _T513] = =(X_I_16[0]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000, 500000000 + i2 < 1000000000
// Flattened:
//              Range    X_T349    X_I_16  
//       i1        80       128         0  
//       i2       128         1         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { i1, i2 }
// Ranges: { 80, 128 }
// Out stride: { 128, 1 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0 }
// Tile size: { 4, 128 }
// Contraction output var shape: fp32(1, 80, 128):(10240, 128, 1):40 KiB
// Computed true ops: 20480
// Computed work groups: 20
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 2048
// Computed mem read: 128
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 5120, 1, 1
__kernel void kernel_c6_sdk_174(__global float* restrict  X_T349, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[1];
  int i1_gid = (get_group_id(0) * 4);
  {
    {
      if ((tid < 1))
      {
        in1_shared[0] = in1[clamp((char)0, (char)0, (char)0)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int i2_tid = (tid % 64);
    int i1_tid = ((tid / 64) % 4);
    for (int i2_lid = 0; i2_lid < 2; i2_lid += 1)
    {
      int i2 = ((64 * i2_lid) + i2_tid);
      float val1 = in1_shared[0];
      agg[i2_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int i2_tid = (tid % 64);
  int i1_tid = ((tid / 64) % 4);
  for (int i2_lid = 0; i2_lid < 2; i2_lid += 1)
  {
    int i2 = ((64 * i2_lid) + i2_tid);
    float LX_T349 = agg[i2_lid];
    int gout_idx = ((128 * (i1_gid + i1_tid)) + i2);
    X_T349[gout_idx] = LX_T349;
  }
}
