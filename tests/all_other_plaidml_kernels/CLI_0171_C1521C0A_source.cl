#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 2 1
// lid: 256 1 1
// Original:
// X_T333[n0, n1, n2, a : _T433, _T434, _T435, _T436] = =(X_T332[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T333[n0, n1, n2, a : _T433, _T434, _T435, _T436] = =(X_T332[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= a < 256, 0 <= a < 288, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 28, 0 <= n2 < 28, 0 <= a < 256 }
// Defracted:
// X_T333[n0, n1, n2, a : _T433, _T434, _T435, _T436] = =(X_T332[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T333    X_T332  
//        a       256         1         1  
//       n1        28      8064      7168  
//       n2        28       288       256  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 256, 28, 28 }
// Out stride: { 1, 8064, 288 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7168, 256 }
// Tile size: { 128, 28, 1 }
// Contraction output var shape: fp32(1, 28, 28, 288):(225792, 8064, 288, 1):882 KiB
// Computed true ops: 401408
// Computed work groups: 56
// Computed inner loops: 1
// Computed shared mem: 14448
// Computed out regs: 16384
// Computed mem read: 14336
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 2, 1
__kernel void kernel_c68_sdk_102(__global float* restrict  X_T333, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3612];
  int a_gid = (get_group_id(1) * 128);
  int n2_gid = get_group_id(0);
  {
    {
      int gbase = (a_gid + (n2_gid * 256));
      int a_tid = (tid % 128);
      int n1_tid = ((tid / 128) % 2);
      for (int n1_lid = 0; n1_lid < 14; n1_lid += 1)
      {
        int n1 = ((2 * n1_lid) + n1_tid);
        int lidx = (a_tid + (129 * n1));
        int gidx = ((gbase + a_tid) + (7168 * n1));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)200703)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n1_tid = ((tid / 32) % 8);
    for (int n1_lid = 0; n1_lid < 4; n1_lid += 1)
    {
      int n1_cond = ((n1_lid < 3) || (n1_tid < 4));
      int n1 = select((int)0, (int)((8 * n1_lid) + n1_tid), (int)n1_cond);
      for (int a_lid = 0; a_lid < 4; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float val1 = in1_shared[(a + (129 * n1))];
        int agg_idx = (a_lid + (n1_lid * 4));
        agg[agg_idx] = select((float)agg[agg_idx], (float)val1, (int)n1_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n1_tid = ((tid / 32) % 8);
  for (int n1_lid = 0; n1_lid < 4; n1_lid += 1)
  {
    int n1_cond = ((n1_lid < 3) || (n1_tid < 4));
    if (n1_cond)
    {
      int n1 = ((8 * n1_lid) + n1_tid);
      for (int a_lid = 0; a_lid < 4; a_lid += 1)
      {
        int a = ((32 * a_lid) + a_tid);
        float LX_T333 = agg[(a_lid + (n1_lid * 4))];
        int gout_idx = (((a_gid + a) + (8064 * n1)) + (288 * n2_gid));
        X_T333[gout_idx] = LX_T333;
      }
    }
  }
}
