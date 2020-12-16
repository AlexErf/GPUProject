#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 256 }
// Out stride: { 1 }
// Elementwise input X_I_83 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(Add)]] X_T242 = add(X_T71, X_I_83)
// Elementwise op: X_T243 = cmp_lt(X_T242, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T244 = cond(X_T243, X_T36, X_T242)
// Elementwise op: [[pid(Sqrt)]] X_T245 = sqrt(X_T244)
// Tile size: { 256 }
// Contraction output var shape: fp32(256):(1):1 KiB
// Computed true ops: 1024
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c124_sdk_60(__global float* restrict  X_T245, __global const float* restrict  X_I_83)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  float LX_I_83 = X_I_83[i1_tid];
  float LX_T242 = (1.0009999641624745e-5f + LX_I_83);
  int LX_T243 = (LX_T242 < (float)0);
  float LX_T244 = select((float)LX_T242, (float)0, (int)LX_T243);
  float LX_T245 = native_sqrt(LX_T244);
  X_T245[i1_tid] = LX_T245;
}
