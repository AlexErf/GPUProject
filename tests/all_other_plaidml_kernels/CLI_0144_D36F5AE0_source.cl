#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 256 }
// Out stride: { 1 }
// Elementwise input X_I_83 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(Add)]] X_T214 = add(X_T43, X_I_83)
// Elementwise op: X_T215 = cmp_lt(X_T214, X_T20)
// Elementwise op: [[pid(Sqrt)]] X_T216 = cond(X_T215, X_T20, X_T214)
// Elementwise op: [[pid(Sqrt)]] X_T217 = sqrt(X_T216)
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
__kernel void kernel_c68_sdk_60(__global float* restrict  X_T217, __global const float* restrict  X_I_83)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  float LX_I_83 = X_I_83[i1_tid];
  float LX_T214 = (1.0009999641624745e-5f + LX_I_83);
  int LX_T215 = (LX_T214 < (float)0);
  float LX_T216 = select((float)LX_T214, (float)0, (int)LX_T215);
  float LX_T217 = native_sqrt(LX_T216);
  X_T217[i1_tid] = LX_T217;
}
