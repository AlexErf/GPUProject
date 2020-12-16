#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 4352 17 1
// lid: 256 1 1
// Original:
// X_T489[n, o0, o1, c : _T671, _T672, _T673, _T674] = =(X_T160[])
// With Index Variables Made Integral:
// X_T489[n, o0, o1, c : _T671, _T672, _T673, _T674] = =(X_T160[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Constraints:{ 0 <= n < 1, 0 <= o0 < 17, 0 <= o1 < 17, 0 <= c < 768, 0 <= 500000000 + c < 1000000000, 0 <= 500000000 + n < 1000000000, 0 <= 500000000 + o0 < 1000000000, 0 <= 500000000 + o1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= n < 1, 0 <= o0 < 17, 0 <= o1 < 17, 0 <= c < 768 }
// Defracted:
// X_T489[n, o0, o1, c : _T671, _T672, _T673, _T674] = =(X_T160[]), 500000000 + c < 1000000000, 500000000 + n < 1000000000, 500000000 + o0 < 1000000000, 500000000 + o1 < 1000000000
// Flattened:
//              Range    X_T489    X_T160  
//        c       768         1         0  
//       o0        17     13056         0  
//       o1        17       768         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { c, o0, o1 }
// Ranges: { 768, 17, 17 }
// Out stride: { 1, 13056, 768 }
// Input 1 offset: 0
// Input 1 stride: { 0, 0, 0 }
// Tile size: { 768, 1, 1 }
// Contraction output var shape: fp32(1, 17, 17, 768):(221952, 13056, 768, 1):867 KiB
// Computed true ops: 443904
// Computed work groups: 289
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 3072
// Computed mem read: 128
// Computed mem write: 3072
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 4352, 17, 1
__kernel void kernel_c56_sdk_161(__global float* restrict  X_T489)
{
  int tid = get_local_id(0);
  float agg[3] = {0, 0, 0, };
  int o1_gid = get_group_id(0);
  int o0_gid = get_group_id(1);
  {
    int c_tid = (tid % 256);
    for (int c_lid = 0; c_lid < 3; c_lid += 1)
    {
      int c = ((256 * c_lid) + c_tid);
      float val1 = 1.0f;
      agg[c_lid] = val1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int c_tid = (tid % 256);
  for (int c_lid = 0; c_lid < 3; c_lid += 1)
  {
    int c = ((256 * c_lid) + c_tid);
    float LX_T489 = agg[c_lid];
    int gout_idx = ((c + (13056 * o0_gid)) + (768 * o1_gid));
    X_T489[gout_idx] = LX_T489;
  }
}
