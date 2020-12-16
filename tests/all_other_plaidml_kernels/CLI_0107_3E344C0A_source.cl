#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 8192 1 1
// lid: 256 1 1
// Original:
// X_T380[i0, i1 : _T554, _T555] = =(X_I_10[i0, i1])
// With Index Variables Made Integral:
// X_T380[i0, i1 : _T554, _T555] = =(X_I_10[i0, i1]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000
// Constraints:{ 0 <= i0 < 128, 0 <= i1 < 128, 0 <= i0 < 128, 0 <= i1 < 512, 0 <= 500000000 + i0 < 1000000000, 0 <= 500000000 + i1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= i0 < 128, 0 <= i1 < 128 }
// Defracted:
// X_T380[i0, i1 : _T554, _T555] = =(X_I_10[i0, i1]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000
// Flattened:
//              Range    X_T380    X_I_10  
//       i0       128       128       512  
//       i1       128         1         1  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { i0, i1 }
// Ranges: { 128, 128 }
// Out stride: { 128, 1 }
// Input 1 offset: 0
// Input 1 stride: { 512, 1 }
// Tile size: { 4, 128 }
// Contraction output var shape: fp32(128, 128):(128, 1):64 KiB
// Computed true ops: 32768
// Computed work groups: 32
// Computed inner loops: 1
// Computed shared mem: 2064
// Computed out regs: 2048
// Computed mem read: 2048
// Computed mem write: 2048
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 8192, 1, 1
__kernel void kernel_c6_sdk_185(__global float* restrict  X_T380, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  __local float in1_shared[516];
  int i0_gid = (get_group_id(0) * 4);
  {
    {
      int gbase = (i0_gid * 512);
      int i1_tid = (tid % 128);
      int i0_tid = ((tid / 128) % 2);
      for (int i0_lid = 0; i0_lid < 2; i0_lid += 1)
      {
        int i0 = ((2 * i0_lid) + i0_tid);
        int lidx = (i1_tid + (129 * i0));
        int gidx = ((gbase + i1_tid) + (512 * i0));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)65535)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int i1_tid = (tid % 64);
    int i0_tid = ((tid / 64) % 4);
    for (int i1_lid = 0; i1_lid < 2; i1_lid += 1)
    {
      int i1 = ((64 * i1_lid) + i1_tid);
      float val1 = in1_shared[(i1 + (129 * i0_tid))];
      agg[i1_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int i1_tid = (tid % 64);
  int i0_tid = ((tid / 64) % 4);
  for (int i1_lid = 0; i1_lid < 2; i1_lid += 1)
  {
    int i1 = ((64 * i1_lid) + i1_tid);
    float LX_T380 = agg[i1_lid];
    int gout_idx = ((128 * (i0_gid + i0_tid)) + i1);
    X_T380[gout_idx] = LX_T380;
  }
}
