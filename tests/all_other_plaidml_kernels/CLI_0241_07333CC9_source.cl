#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 3 21
// lid: 256 1 1
// Original:
// X_T2880[n, o0, o1, c : _T4579, _T4580, _T4581, _T4582] = =(X_T100[])
// With Index Variables Made Integral:
// X_T2880[n, o0, o1, c : _T4579, _T4580, _T4581, _T4582] = =(X_T100[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 23, 0 <= o1 < 23, 0 <= c < 672, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 23, 0 <= o1 < 23, 0 <= c < 672 }
// Defracted:
// X_T2880[n, o0, o1, c : _T4579, _T4580, _T4581, _T4582] = =(X_T100[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range   X_T2880    X_T100  
//        c       672         1         0  
//       o0        23     15456         0  
//       o1        23       672         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 672, 23, 23 }
// Out stride: { 1, 15456, 672 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 32, 8, 8 }
// Contraction output var shape: fp32(1, 23, 23, 672):(355488, 15456, 672, 1):1388.62 KiB
// Computed true ops: 710976
// Computed work groups: 189
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 8192
// Computed mem read: 128
// Computed mem write: 8192
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 3, 21
__kernel void kernel_c42_sdk_1113(__global float* restrict  X_T2880)
{
  int tid = get_local_id(0);
  float agg[8] = {0, 0, 0, 0, 0, 0, 0, 0, };
  int c_gid = (get_group_id(2) * 32);
  int o1_gid = (get_group_id(0) * 8);
  int o0_gid = (get_group_id(1) * 8);
  {
    int c_tid = (tid % 32);
    int o1_tid = ((tid / 32) % 4);
    int o0_tid = ((tid / 128) % 2);
    for (int o1_lid = 0; o1_lid < 2; o1_lid += 1)
    {
      int o1 = ((4 * o1_lid) + o1_tid);
      for (int o0_lid = 0; o0_lid < 4; o0_lid += 1)
      {
        int o0 = ((2 * o0_lid) + o0_tid);
        float val1 = 1.0f;
        int agg_idx = (o1_lid + (o0_lid * 2));
        agg[agg_idx] = val1;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o1_tid = ((tid / 32) % 4);
  int o0_tid = ((tid / 128) % 2);
  for (int o1_lid = 0; o1_lid < 2; o1_lid += 1)
  {
    int o1_cond = ((o1_lid < 1) || ((o1_gid != 16) || (o1_tid < 3)));
    if (o1_cond)
    {
      int o1 = ((4 * o1_lid) + o1_tid);
      for (int o0_lid = 0; o0_lid < 4; o0_lid += 1)
      {
        int o0_cond = ((o0_lid < 3) || ((o0_gid != 16) || (o0_tid < 1)));
        if (o0_cond)
        {
          int o0 = ((2 * o0_lid) + o0_tid);
          float LX_T2880 = agg[(o1_lid + (o0_lid * 2))];
          int gout_idx = (((c_gid + c_tid) + (15456 * (o0_gid + o0))) + (672 * (o1_gid + o1)));
          X_T2880[gout_idx] = LX_T2880;
        }
      }
    }
  }
}
