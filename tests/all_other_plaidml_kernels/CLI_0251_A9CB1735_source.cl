#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 7 1
// lid: 256 1 1
// Original:
// X_T2287[n, o0, o1, c : _T3621, _T3622, _T3623, _T3624] = =(X_T97[])
// With Index Variables Made Integral:
// X_T2287[n, o0, o1, c : _T3621, _T3622, _T3623, _T3624] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 7, 0 <= o1 < 7, 0 <= c < 176, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 7, 0 <= o1 < 7, 0 <= c < 176 }
// Defracted:
// X_T2287[n, o0, o1, c : _T3621, _T3622, _T3623, _T3624] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range   X_T2287     X_T97  
//        c       176         1         0  
//       o0         7      1232         0  
//       o1         7       176         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 176, 7, 7 }
// Out stride: { 1, 1232, 176 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 64, 7, 1 }
// Contraction output var shape: fp32(1, 7, 7, 176):(8624, 1232, 176, 1):33.6875 KiB
// Computed true ops: 17248
// Computed work groups: 21
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 2048
// Computed mem read: 128
// Computed mem write: 1792
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 7, 1
__kernel void kernel_c42_sdk_874(__global float* restrict  X_T2287)
{
  int tid = get_local_id(0);
  float agg[2] = {0, 0, };
  int c_gid = (get_group_id(0) * 64);
  int o1_gid = get_group_id(1);
  {
    int c_tid = (tid % 32);
    int o0_tid = ((tid / 32) % 8);
    for (int c_lid = 0; c_lid < 2; c_lid += 1)
    {
      int c = ((32 * c_lid) + c_tid);
      int o0_cond = (o0_tid < 7);
      int o0 = select((int)0, (int)o0_tid, (int)o0_cond);
      float val1 = 1.0f;
      agg[c_lid] = select((float)agg[c_lid], (float)val1, (int)o0_cond);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o0_tid = ((tid / 32) % 8);
  for (int c_lid = 0; c_lid < 2; c_lid += 1)
  {
    int c_cond = ((c_lid < 1) || ((c_gid != 128) || (c_tid < 16)));
    if (c_cond)
    {
      int c = ((32 * c_lid) + c_tid);
      int o0_cond = (o0_tid < 7);
      if (o0_cond)
      {
        float LX_T2287 = agg[c_lid];
        int gout_idx = (((c_gid + c) + (1232 * o0_tid)) + (176 * o1_gid));
        X_T2287[gout_idx] = LX_T2287;
      }
    }
  }
}
