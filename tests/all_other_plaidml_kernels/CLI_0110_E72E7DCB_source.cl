#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Original:
// X_T395[i0 : _T575] = =(X_I_14[256 + i0])
// With Index Variables Made Integral:
// X_T395[i0 : _T575] = =(X_I_14[256 + i0]), 500000000 + i0 < 1000000000
// Constraints:{ 0 <= i0 < 128, 0 <= 256 + i0 < 512, 0 <= 500000000 + i0 < 1000000000 }
// Merged Parallel Constraints:{ 0 <= i0 < 128 }
// Defracted:
// X_T395[i0 : _T575] = =(X_I_14[256 + i0]), 500000000 + i0 < 1000000000
// Flattened:
//              Range    X_T395    X_I_14  
//       i0       128         1         1  
//      off                   0       256  
//      vec                   1         1  
// 
// Names: { i0 }
// Ranges: { 128 }
// Out stride: { 1 }
// Input 1 offset: 256
// Input 1 stride: { 1 }
// Elementwise input X_T394 shape: fp32(1, 128):(128, 1):512 bytes
// Elementwise op: [[pid(Add)]] X_T396 = add(X_T394, X_T395)
// Tile size: { 128 }
// Contraction output var shape: fp32(128):(1):512 bytes
// Computed true ops: 384
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 512
// Computed out regs: 1024
// Computed mem read: 528
// Computed mem write: 1024
// Computed operations: 128
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c6_sdk_192(__global float* restrict  X_T395, __global float* restrict  X_T396, __global const float* restrict  in1, __global const float* restrict  X_T394)
{
  in1 = (in1 + 256);
  int tid = get_local_id(0);
  float agg[1] = {0, };
  __local float in1_shared[128];
  {
    {
      int i0_tid = (tid % 128);
      in1_shared[i0_tid] = in1[clamp((int)i0_tid, (int)-256, (int)255)];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int i0_tid = (tid % 128);
    float val1 = in1_shared[i0_tid];
    agg[0] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  int i0_tid = (tid % 128);
  float LX_T395 = agg[0];
  float LX_T394 = X_T394[i0_tid];
  float LX_T396 = (LX_T394 + LX_T395);
  X_T395[i0_tid] = LX_T395;
  X_T396[i0_tid] = LX_T396;
}
