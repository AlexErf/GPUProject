#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Original:
// X_T61[i0, i1 : _T79, _T80] = =(X_T25[i0, 71, i1])
// With Index Variables Made Integral:
// X_T61[i0, i1 : _T79, _T80] = =(X_T25[i0, 71, i1]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000
// Constraints:{ 0 <= i0 < 1, 0 <= i0 < 1, 0 <= 71 < 80, 0 <= i1 < 128, 0 <= i1 < 128, 0 <= 500000000 + i0 < 1000000000, 0 <= 500000000 + i1 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= i0 < 1, 0 <= i1 < 128 }
// Defracted:
// X_T61[i0, i1 : _T79, _T80] = =(X_T25[i0, 71, i1]), 500000000 + i0 < 1000000000, 500000000 + i1 < 1000000000
// Flattened:
//              Range     X_T61     X_T25  
//       i1       128         1         1  
//      off                   0      9088  
//      vec                   1         1  
// 
// Names: { i1 }
// Ranges: { 128 }
// Out stride: { 1 }
// Input 1 offset: 9088
// Input 1 stride: { 1 }
// Elementwise input X_T27 shape: fp32(1, 128):(128, 1):512 bytes
// Elementwise op: [[pid(Mul)]] X_T62 = mul(X_T61, X_T27)
// Tile size: { 128 }
// Contraction output var shape: fp32(1, 128):(128, 1):512 bytes
// Computed true ops: 384
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 512
// Computed out regs: 1024
// Computed mem read: 528
// Computed mem write: 512
// Computed operations: 128
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c6_sdk_30(__global float* restrict  X_T62, __global const float* restrict  in1, __global const float* restrict  X_T27)
{
  in1 = (in1 + 9088);
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[128];
  {
    {
      int i1_tid = (tid % 128);
      in1_shared[i1_tid] = in1[clamp((int)i1_tid, (int)-9088, (int)1151)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int i1_tid = (tid % 128);
    float val1 = in1_shared[i1_tid];
    agg[0] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int i1_tid = (tid % 128);
  float LX_T61 = agg[0];
  float LX_T27 = X_T27[i1_tid];
  float LX_T62 = (LX_T61 * LX_T27);
  X_T62[i1_tid] = LX_T62;
}
