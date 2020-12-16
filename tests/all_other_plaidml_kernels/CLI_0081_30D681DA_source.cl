#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Original:
// X_T391[n, d0, d1, c : _T438, _T439, _T440, _T441] = =(X_T389[n, d0, d1, c])
// With Index Variables Made Integral:
// X_T391[n, d0, d1, c : _T438, _T439, _T440, _T441] = =(X_T389[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= d0 < 14, 0 <= d1 < 14, 0 <= d0 < 15, 0 <= d1 < 15, 0 <= c < 512, 0 <= c < 512, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= d0 < 14, 0 <= d1 < 14, 0 <= c < 512 }
// Defracted:
// X_T391[n, d0, d1, c : _T438, _T439, _T440, _T441] = =(X_T389[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range    X_T391    X_T389  
//        c       512         1         1  
//       d0        14      7680      7168  
//       d1        14       512       512  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 7168, 14 }
// Out stride: { 1, 7680 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7168 }
// Tile size: { 256, 14 }
// Contraction output var shape: fp32(1, 15, 15, 512):(115200, 7680, 512, 1):450 KiB
// Computed true ops: 200704
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 14392
// Computed out regs: 16384
// Computed mem read: 14336
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c25_sdk_98(__global float* restrict  X_T391, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3598];
  int d1_c_gid = (get_group_id(0) * 256);
  {
    {
      int d1_c_tid = (tid % 256);
      for (int d0_lid = 0; d0_lid < 14; d0_lid += 1)
      {
        int lidx = (d1_c_tid + (257 * d0_lid));
        int gidx = ((d1_c_gid + d1_c_tid) + (7168 * d0_lid));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)100351)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int d1_c_tid = (tid % 32);
    int d0_tid = ((tid / 32) % 8);
    for (int d0_lid = 0; d0_lid < 2; d0_lid += 1)
    {
      int d0_cond = ((d0_lid < 1) || (d0_tid < 6));
      int d0 = select((int)0, (int)((8 * d0_lid) + d0_tid), (int)d0_cond);
      for (int d1_c_lid = 0; d1_c_lid < 8; d1_c_lid += 1)
      {
        int d1_c = ((32 * d1_c_lid) + d1_c_tid);
        float val1 = in1_shared[(d1_c + (257 * d0))];
        int agg_idx = (d1_c_lid + (d0_lid * 8));
        agg[agg_idx] = select((float)agg[agg_idx], (float)val1, (int)d0_cond);
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int d1_c_tid = (tid % 32);
  int d0_tid = ((tid / 32) % 8);
  for (int d0_lid = 0; d0_lid < 2; d0_lid += 1)
  {
    int d0_cond = ((d0_lid < 1) || (d0_tid < 6));
    if (d0_cond)
    {
      int d0 = ((8 * d0_lid) + d0_tid);
      for (int d1_c_lid = 0; d1_c_lid < 8; d1_c_lid += 1)
      {
        int d1_c = ((32 * d1_c_lid) + d1_c_tid);
        float LX_T391 = agg[(d1_c_lid + (d0_lid * 8))];
        int gout_idx = ((d1_c_gid + d1_c) + (7680 * d0));
        X_T391[gout_idx] = LX_T391;
      }
    }
  }
}
