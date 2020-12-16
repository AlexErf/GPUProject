#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 128 1 1
// lid: 128 1 1
// Names: { i1 }
// Ranges: { 128 }
// Out stride: { 1 }
// Elementwise input X_I_264 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(Add)]] X_T174 = add(X_T37, X_I_264)
// Elementwise op: X_T175 = cmp_lt(X_T174, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T176 = cond(X_T175, X_T3, X_T174)
// Elementwise op: [[pid(Sqrt)]] X_T177 = sqrt(X_T176)
// Tile size: { 128 }
// Contraction output var shape: fp32(128):(1):512 bytes
// Computed true ops: 512
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 16
// Computed mem write: 512
// Computed operations: 128
// Computed rollups: 0
// Computed threads used: 128
// lwork = 128, 1, 1
// gwork = 128, 1, 1
__kernel void kernel_c29_sdk_37(__global float* restrict  X_T177, __global const float* restrict  X_I_264)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 128);
  float LX_I_264 = X_I_264[i1_tid];
  float LX_T174 = (0.0010000000474974513f + LX_I_264);
  int LX_T175 = (LX_T174 < (float)0);
  float LX_T176 = select((float)LX_T174, (float)0, (int)LX_T175);
  float LX_T177 = native_sqrt(LX_T176);
  X_T177[i1_tid] = LX_T177;
}
