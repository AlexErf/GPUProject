#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 57344 1 1
// lid: 256 1 1
// Original:
// X_T115[n, d0, d1, c : _T86, _T87, _T88, _T89] = =(X_T113[n, d0, d1, c])
// With Index Variables Made Integral:
// X_T115[n, d0, d1, c : _T86, _T87, _T88, _T89] = =(X_T113[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= n < 1, 0 <= c < 64, 0 <= c < 64, 0 <= d0 < 112, 0 <= d1 < 112, 0 <= d0 < 113, 0 <= d1 < 113, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + d0 < 1000000000, 0 <= 500000000 + d1 < 1000000000, 0 <= 500000000 + n < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= c < 64, 0 <= d0 < 112, 0 <= d1 < 112 }
// Defracted:
// X_T115[n, d0, d1, c : _T86, _T87, _T88, _T89] = =(X_T113[n, d0, d1, c]), 500000000 + c < 1000000000, 500000000 + d0 < 1000000000, 500000000 + d1 < 1000000000, 500000000 + n < 1000000000
// Flattened:
//              Range    X_T115    X_T113  
//        c        64         1         1  
//       d0       112      7232      7168  
//       d1       112        64        64  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { d1_c, d0 }
// Ranges: { 7168, 112 }
// Out stride: { 1, 7232 }
// Input 1 offset: 0
// Input 1 stride: { 1, 7168 }
// Tile size: { 32, 112 }
// Contraction output var shape: fp32(1, 113, 113, 64):(817216, 7232, 64, 1):3192.25 KiB
// Computed true ops: 1605632
// Computed work groups: 224
// Computed inner loops: 1
// Computed shared mem: 14464
// Computed out regs: 14336
// Computed mem read: 14336
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 57344, 1, 1
__kernel void kernel_c25_sdk_25(__global float* restrict  X_T115, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, };
  __local float in1_shared[3616];
  int d1_c_gid = (get_group_id(0) * 32);
  {
    {
      int d1_c_tid = (tid % 32);
      int d0_tid = ((tid / 32) % 8);
      for (int d0_lid = 0; d0_lid < 14; d0_lid += 1)
      {
        int d0 = ((8 * d0_lid) + d0_tid);
        int lidx = ((113 * d1_c_tid) + d0);
        int gidx = ((d1_c_gid + d1_c_tid) + (7168 * d0));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)802815)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int d1_c_tid = (tid % 32);
    int d0_tid = ((tid / 32) % 8);
    for (int d0_lid = 0; d0_lid < 14; d0_lid += 1)
    {
      int d0 = ((8 * d0_lid) + d0_tid);
      float val1 = in1_shared[((113 * d1_c_tid) + d0)];
      agg[d0_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int d1_c_tid = (tid % 32);
  int d0_tid = ((tid / 32) % 8);
  for (int d0_lid = 0; d0_lid < 14; d0_lid += 1)
  {
    int d0 = ((8 * d0_lid) + d0_tid);
    float LX_T115 = agg[d0_lid];
    int gout_idx = ((d1_c_gid + d1_c_tid) + (7232 * d0));
    X_T115[gout_idx] = LX_T115;
  }
}
