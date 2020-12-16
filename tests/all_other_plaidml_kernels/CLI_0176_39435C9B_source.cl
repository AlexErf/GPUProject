#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 224 }
// Out stride: { 1 }
// Elementwise input X_I_734 shape: fp32(224):(1):896 bytes
// Elementwise op: [[pid(Add)]] X_T2029 = add(X_T33, X_I_734)
// Elementwise op: X_T2030 = cmp_lt(X_T2029, X_T4)
// Elementwise op: [[pid(Sqrt)]] X_T2031 = cond(X_T2030, X_T4, X_T2029)
// Elementwise op: [[pid(Sqrt)]] X_T2032 = sqrt(X_T2031)
// Tile size: { 224 }
// Contraction output var shape: fp32(224):(1):896 bytes
// Computed true ops: 896
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 28
// Computed mem write: 896
// Computed operations: 224
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c51_sdk_663(__global float* restrict  X_T2032, __global const float* restrict  X_I_734)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  int i1_cond = (i1_tid < 224);
  if (i1_cond)
  {
    float LX_I_734 = X_I_734[i1_tid];
    float LX_T2029 = (0.0010000000474974513f + LX_I_734);
    int LX_T2030 = (LX_T2029 < (float)0);
    float LX_T2031 = select((float)LX_T2029, (float)0, (int)LX_T2030);
    float LX_T2032 = native_sqrt(LX_T2031);
    X_T2032[i1_tid] = LX_T2032;
  }
}
