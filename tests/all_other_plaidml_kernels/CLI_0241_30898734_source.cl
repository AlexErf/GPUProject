#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 15 1
// lid: 256 1 1
// Original:
// X_T2188[n, o0, o1, c : _T3467, _T3468, _T3469, _T3470] = =(X_T97[])
// With Index Variables Made Integral:
// X_T2188[n, o0, o1, c : _T3467, _T3468, _T3469, _T3470] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 15, 0 <= o1 < 15, 0 <= c < 176, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 15, 0 <= o1 < 15, 0 <= c < 176 }
// Defracted:
// X_T2188[n, o0, o1, c : _T3467, _T3468, _T3469, _T3470] = =(X_T97[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range   X_T2188     X_T97  
//        c       176         1         0  
//       o0        15      2640         0  
//       o1        15       176         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 176, 15, 15 }
// Out stride: { 1, 2640, 176 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 64, 15, 1 }
// Contraction output var shape: fp32(1, 15, 15, 176):(39600, 2640, 176, 1):154.688 KiB
// Computed true ops: 79200
// Computed work groups: 45
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 4096
// Computed mem read: 128
// Computed mem write: 3840
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 15, 1
__kernel void kernel_c42_sdk_837(__global float* restrict  X_T2188)
{
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  int c_gid = (get_group_id(0) * 64);
  int o1_gid = get_group_id(1);
  {
    int c_tid = (tid % 32);
    int o0_tid = ((tid / 32) % 8);
    for (int c_lid = 0; c_lid < 2; c_lid += 1)
    {
      int c = ((32 * c_lid) + c_tid);
      for (int o0_lid = 0; o0_lid < 2; o0_lid += 1)
      {
        int o0_cond = ((o0_lid < 1) || (o0_tid < 7));
        int o0 = select((int)0, (int)((8 * o0_lid) + o0_tid), (int)o0_cond);
        float val1 = 1.0f;
        int agg_idx = (c_lid + (o0_lid * 2));
        agg[agg_idx] = select((float)agg[agg_idx], (float)val1, (int)o0_cond);
      }
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
      for (int o0_lid = 0; o0_lid < 2; o0_lid += 1)
      {
        int o0_cond = ((o0_lid < 1) || (o0_tid < 7));
        if (o0_cond)
        {
          int o0 = ((8 * o0_lid) + o0_tid);
          float LX_T2188 = agg[(c_lid + (o0_lid * 2))];
          int gout_idx = (((c_gid + c) + (2640 * o0)) + (176 * o1_gid));
          X_T2188[gout_idx] = LX_T2188;
        }
      }
    }
  }
}
