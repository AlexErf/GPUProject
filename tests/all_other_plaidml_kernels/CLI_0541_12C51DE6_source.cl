#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 14 1
// lid: 256 1 1
// Original:
// X_T1581[n0, n1, n2, a : _T2240, _T2241, _T2242, _T2243] = =(X_T1580[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T1581[n0, n1, n2, a : _T2240, _T2241, _T2242, _T2243] = =(X_T1580[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 1536, 0 <= a < 1568, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 14, 0 <= n2 < 14, 0 <= a < 1536 }
// Defracted:
// X_T1581[n0, n1, n2, a : _T2240, _T2241, _T2242, _T2243] = =(X_T1580[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range   X_T1581   X_T1580  
//        a      1536         1         1  
//       n1        14     21952     21504  
//       n2        14      1568      1536  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 1536, 14, 14 }
// Out stride: { 1, 21952, 1568 }
// Input 1 offset: 0
// Input 1 stride: { 1, 21504, 1536 }
// Tile size: { 256, 14, 1 }
// Contraction output var shape: fp32(1, 14, 14, 1568):(307328, 21952, 1568, 1):1200.5 KiB
// Computed true ops: 602112
// Computed work groups: 84
// Computed inner loops: 1
// Computed shared mem: 14392
// Computed out regs: 16384
// Computed mem read: 14336
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 14, 1
__kernel void kernel_c124_sdk_540(__global float* restrict  X_T1581, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3598];
  int a_gid = (get_group_id(0) * 256);
  int n2_gid = get_group_id(1);
  {
    {
      int gbase = (a_gid + (n2_gid * 1536));
      int a_tid = (tid % 256);
      for (int n1_lid = 0; n1_lid < 14; n1_lid += 1)
      {
        int lidx = (a_tid + (257 * n1_lid));
        int gidx = ((gbase + a_tid) + (21504 * n1_lid));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)301055)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n1_tid = ((tid / 32) % 8);
    for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
    {
      int n1_cond = ((n1_lid < 1) || (n1_tid < 6));
      int n1 = select((int)0, (int)((8 * n1_lid) + n1_tid), (int)n1_cond);
      for (int a_lid = 0; a_lid < 8; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float val1 = in1_shared[(a + (257 * n1))];
        int agg_idx = (a_lid + (n1_lid * 8));
        agg[agg_idx] = select((float)agg[agg_idx], (float)val1, (int)n1_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n1_tid = ((tid / 32) % 8);
  for (int n1_lid = 0; n1_lid < 2; n1_lid += 1)
  {
    int n1_cond = ((n1_lid < 1) || (n1_tid < 6));
    if (n1_cond)
    {
      int n1 = ((8 * n1_lid) + n1_tid);
      for (int a_lid = 0; a_lid < 8; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float LX_T1581 = agg[(a_lid + (n1_lid * 8))];
        int gout_idx = (((a_gid + a) + (21952 * n1)) + (1568 * n2_gid));
        X_T1581[gout_idx] = LX_T1581;
      }
    }
  }
}
