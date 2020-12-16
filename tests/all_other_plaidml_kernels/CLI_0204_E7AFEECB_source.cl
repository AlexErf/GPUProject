#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 8 1
// lid: 256 1 1
// Original:
// X_T945[n0, n1, n2, 320 + a : _T1331, _T1332, _T1333, _T1334] = =(X_T944[n0, n1, n2, a])
// With Index Variables Made Integral:
// X_T945[n0, n1, n2, 320 + a : _T1331, _T1332, _T1333, _T1334] = =(X_T944[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Constraints:{ 0 <= n0 < 1, 0 <= n0 < 1, 0 <= n1 < 8, 0 <= n2 < 8, 0 <= n1 < 8, 0 <= n2 < 8, 0 <= a < 768, 0 <= 320 + a < 2048, 0 <= 500000000 + a < 1000000000, 0 <= 500000000 + n0 < 1000000000, 0 <= 500000000 + n1 < 1000000000, 0 <= 500000000 + n2 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n0 < 1, 0 <= n1 < 8, 0 <= n2 < 8, 0 <= a < 768 }
// Defracted:
// X_T945[n0, n1, n2, 320 + a : _T1331, _T1332, _T1333, _T1334] = =(X_T944[n0, n1, n2, a]), 500000000 + a < 1000000000, 500000000 + n0 < 1000000000, 500000000 + n1 < 1000000000, 500000000 + n2 < 1000000000
// Flattened:
//              Range    X_T945    X_T944  
//        a       768         1         1  
//       n1         8     16384      6144  
//       n2         8      2048       768  
//      off                 320         0  
//      vec                   1         1  
// 
// Names: { a, n1, n2 }
// Ranges: { 768, 8, 8 }
// Out stride: { 1, 16384, 2048 }
// Input 1 offset: 0
// Input 1 stride: { 1, 6144, 768 }
// Tile size: { 128, 1, 8 }
// Contraction output var shape: fp32(1, 8, 8, 2048):(131072, 16384, 2048, 1):512 KiB
// Computed true ops: 98304
// Computed work groups: 48
// Computed inner loops: 1
// Computed shared mem: 4128
// Computed out regs: 4096
// Computed mem read: 4096
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 8, 1
__kernel void kernel_c56_sdk_322(__global float* restrict  X_T945, __global const float* restrict  in1)
{
  X_T945 = (X_T945 + 320);
  int tid = get_local_id(0);
  float agg[4] = {0, 0, 0, 0, };
  __local float in1_shared[1032];
  int a_gid = (get_group_id(0) * 128);
  int n1_gid = get_group_id(1);
  {
    {
      int gbase = (a_gid + (n1_gid * 6144));
      int a_tid = (tid % 128);
      int n2_n1_tid = ((tid / 128) % 2);
      for (int n2_n1_lid = 0; n2_n1_lid < 4; n2_n1_lid += 1)
      {
        int n2_n1 = ((2 * n2_n1_lid) + n2_n1_tid);
        int lidx = (a_tid + (129 * n2_n1));
        int gidx = ((gbase + a_tid) + (768 * n2_n1));
        in1_shared[lidx] = in1[clamp((int)gidx, (int)0, (int)49151)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int a_tid = (tid % 32);
    int n2_tid = ((tid / 32) % 8);
    for (int a_lid = 0; a_lid < 4; a_lid += 1)
    {
      int a = ((32 * a_lid) + a_tid);
      float val1 = in1_shared[(a + (129 * n2_tid))];
      agg[a_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int a_tid = (tid % 32);
  int n2_tid = ((tid / 32) % 8);
  for (int a_lid = 0; a_lid < 4; a_lid += 1)
  {
    int a = ((32 * a_lid) + a_tid);
    float LX_T945 = agg[a_lid];
    int gout_idx = (((a_gid + a) + (16384 * n1_gid)) + (2048 * n2_tid));
    X_T945[gout_idx] = LX_T945;
  }
}
