#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Original:
// X_T27[i0, i1 : _T28, _T29] = =(X_I_13[0])
// With Index Variables Made Integral:
// X_T27[i0, i1 : _T28, _T29] = =(X_I_13[0]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000
// Constraints:{ 0 <= i0 < 1, 0 <= 0 < 1, 0 <= i1 < 128, 0 <= 500000000 + i0 < 1000000000, 0 <= 500000000 + i1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= i0 < 1, 0 <= i1 < 128 }
// Defracted:
// X_T27[i0, i1 : _T28, _T29] = =(X_I_13[0]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000
// Flattened:
//              Range     X_T27    X_I_13  
//       i1       128         1         0  
//      off                   0         0  
//      vec                   1         1  
// 
// Names: { i1 }
// Ranges: { 128 }
// Out stride: { 1 }
// Input 1 offset: 0
// Input 1 stride: { 0 }
// Elementwise input X_T26 shape: fp32(1, 128):(128, 1):512 bytes
// Elementwise op: [[pid(Mul)]] X_T28 = mul(X_T26, X_T27)
// Tile size: { 128 }
// Contraction output var shape: fp32(1, 128):(128, 1):512 bytes
// Computed true ops: 384
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 4
// Computed out regs: 1024
// Computed mem read: 144
// Computed mem write: 1024
// Computed operations: 128
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c6_sdk_12(__global float* restrict  X_T27, __global float* restrict  X_T28, __global const float* restrict  in1, __global const float* restrict  X_T26)
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
    int i1_tid = (tid % 128);
    float val1 = in1_shared[0];
    agg[0] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int i1_tid = (tid % 128);
  float LX_T27 = agg[0];
  float LX_T26 = X_T26[i1_tid];
  float LX_T28 = (LX_T26 * LX_T27);
  X_T27[i1_tid] = LX_T27;
  X_T28[i1_tid] = LX_T28;
}
