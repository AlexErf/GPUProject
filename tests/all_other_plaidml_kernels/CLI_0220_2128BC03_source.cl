#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 8 1
// lid: 256 1 1
// Original:
// X_T1100[n, o0, o1, c : _T1579, _T1580, _T1581, _T1582] = =(X_T160[])
// With Index Variables Made Integral:
// X_T1100[n, o0, o1, c : _T1579, _T1580, _T1581, _T1582] = =(X_T160[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 8, 0 <= o1 < 8, 0 <= c < 2048, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 8, 0 <= o1 < 8, 0 <= c < 2048 }
// Defracted:
// X_T1100[n, o0, o1, c : _T1579, _T1580, _T1581, _T1582] = =(X_T160[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range   X_T1100    X_T160  
//        c      2048         1         0  
//       o0         8     16384         0  
//       o1         8      2048         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 2048, 8, 8 }
// Out stride: { 1, 16384, 2048 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 512, 1, 8 }
// Contraction output var shape: fp32(1, 8, 8, 2048):(131072, 16384, 2048, 1):512 KiB
// Computed true ops: 262144
// Computed work groups: 32
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 16384
// Computed mem read: 128
// Computed mem write: 16384
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1024, 8, 1
__kernel void kernel_c56_sdk_385(__global float* restrict  X_T1100)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  int c_gid = (get_group_id(0) * 512);
  int o0_gid = get_group_id(1);
  {
    int c_tid = (tid % 32);
    int o1_tid = ((tid / 32) % 8);
    for (int c_lid = 0; c_lid < 16; c_lid += 1)
    {
      int c = ((32 * c_lid) + c_tid);
      float val1 = 1.0f;
      agg[c_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o1_tid = ((tid / 32) % 8);
  for (int c_lid = 0; c_lid < 16; c_lid += 1)
  {
    int c = ((32 * c_lid) + c_tid);
    float LX_T1100 = agg[c_lid];
    int gout_idx = (((c_gid + c) + (16384 * o0_gid)) + (2048 * o1_tid));
    X_T1100[gout_idx] = LX_T1100;
  }
}
