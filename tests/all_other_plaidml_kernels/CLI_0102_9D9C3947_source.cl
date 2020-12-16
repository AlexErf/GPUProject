#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Original:
// X_T352[i0 + t0, i1 + t1 : _T517, _T518] = =(X_T351[i0, i1]), t0 < 1, t1 < 128 no_defract
// With Index Variables Made Integral:
// X_T352[i0 + t0, i1 + t1 : _T517, _T518] = =(X_T351[i0, i1]), t0 < 1, t1 < 128, 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000, 500000000 + t0 < 1000000000, 500000000 + t1 < 1000000000 no_defract
// Constraints:{ 0 <= t0 < 1, 0 <= i0 + t0 < 1, 0 <= i0 < 1, 0 <= i1 < 1, 0 <= t1 < 128, 0 <= i1 + t1 < 128, 0 <= 500000000 + i0 < 1000000000, 0 <= 500000000 + i1 < 1000000000, 0 <= 500000000 + t0 < 1000000000, 0 <= 500000000 + t1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= t0 < 1, 0 <= i0 + t0 < 1, 0 <= i0 < 1, 0 <= i1 < 1, 0 <= t1 < 128, 0 <= i1 + t1 < 128 }
// Defracted:
// X_T352[i0 + t0, i1 + t1 : _T517, _T518] = =(X_T351[i0, i1]), t0 < 1, t1 < 128, 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000, 500000000 + t0 < 1000000000, 500000000 + t1 < 1000000000 no_defract
// Flattened:
//              Range    X_T352    X_T351  
//       t1       128         1         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { t1 }
// Ranges: { 128 }
// Out stride: { 1 }
// Input 1 offset: 0
// Input 1 stride: { 0 }
// Tile size: { 128 }
// Contraction output var shape: fp32(1, 128):(128, 1):512 bytes
// Computed true ops: 256
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 1024
// Computed mem read: 128
// Computed mem write: 512
// Computed operations: 128
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c6_sdk_177(__global float* restrict  X_T352, __global const float* restrict  in1)
{
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[1];
  {
    {
      if ((tid < 1))
      {
        in1_shared[0] = in1[clamp((char)0, (char)0, (char)0)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int t1_tid = (tid % 128);
    float val1 = in1_shared[0];
    agg[0] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int t1_tid = (tid % 128);
  float LX_T352 = agg[0];
  X_T352[t1_tid] = LX_T352;
}
