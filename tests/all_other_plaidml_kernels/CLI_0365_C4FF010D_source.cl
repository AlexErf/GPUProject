#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 2 1
// lid: 256 1 1
// Original:
// X_T1166[n, o0, o1, c : _T1668, _T1669, _T1670, _T1671] = =(X_T225[])
// With Index Variables Made Integral:
// X_T1166[n, o0, o1, c : _T1668, _T1669, _T1670, _T1671] = =(X_T225[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 14, 0 <= o1 < 14, 0 <= c < 512, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 14, 0 <= o1 < 14, 0 <= c < 512 }
// Defracted:
// X_T1166[n, o0, o1, c : _T1668, _T1669, _T1670, _T1671] = =(X_T225[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range   X_T1166    X_T225  
//        c       512         1         0  
//       o0        14      7168         0  
//       o1        14       512         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 512, 14, 14 }
// Out stride: { 1, 7168, 512 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 256, 14, 1 }
// Contraction output var shape: fp32(1, 14, 14, 512):(100352, 7168, 512, 1):392 KiB
// Computed true ops: 200704
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 16384
// Computed mem read: 128
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 2, 1
__kernel void kernel_c68_sdk_400(__global float* restrict  X_T1166)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  int c_gid = (get_group_id(1) * 256);
  int o1_gid = get_group_id(0);
  {
    int c_tid = (tid % 32);
    int o0_tid = ((tid / 32) % 8);
    for (int o0_lid = 0; o0_lid < 2; o0_lid += 1)
    {
      int o0_cond = ((o0_lid < 1) || (o0_tid < 6));
      int o0 = select((int)0, (int)((8 * o0_lid) + o0_tid), (int)o0_cond);
      for (int c_lid = 0; c_lid < 8; c_lid += 1)
      {
        int c = ((32 * c_lid) + c_tid);
        float val1 = 1.0f;
        int agg_idx = (c_lid + (o0_lid * 8));
        agg[agg_idx] = select((float)agg[agg_idx], (float)val1, (int)o0_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 32);
  int o0_tid = ((tid / 32) % 8);
  for (int o0_lid = 0; o0_lid < 2; o0_lid += 1)
  {
    int o0_cond = ((o0_lid < 1) || (o0_tid < 6));
    if (o0_cond)
    {
      int o0 = ((8 * o0_lid) + o0_tid);
      for (int c_lid = 0; c_lid < 8; c_lid += 1)
      {
        int c = ((32 * c_lid) + c_tid);
        float LX_T1166 = agg[(c_lid + (o0_lid * 8))];
        int gout_idx = (((c_gid + c) + (7168 * o0)) + (512 * o1_gid));
        X_T1166[gout_idx] = LX_T1166;
      }
    }
  }
}
