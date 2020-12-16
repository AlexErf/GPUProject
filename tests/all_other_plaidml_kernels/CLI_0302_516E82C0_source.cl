#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 512 }
// Out stride: { 1 }
// Elementwise input X_I_204 shape: fp32(512):(1):2 KiB
// Elementwise op: [[pid(Add)]] X_T563 = add(X_T71, X_I_204)
// Elementwise op: X_T564 = cmp_lt(X_T563, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T565 = cond(X_T564, X_T36, X_T563)
// Elementwise op: [[pid(Sqrt)]] X_T566 = sqrt(X_T565)
// Tile size: { 256 }
// Contraction output var shape: fp32(512):(1):2 KiB
// Computed true ops: 2048
// Computed work groups: 2
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 1, 1
__kernel void kernel_c124_sdk_174(__global float* restrict  X_T566, __global const float* restrict  X_I_204)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int gout_idx = (i1_gid + i1_tid);
  float LX_I_204 = X_I_204[gout_idx];
  float LX_T563 = (1.0009999641624745e-5f + LX_I_204);
  int LX_T564 = (LX_T563 < (float)0);
  float LX_T565 = select((float)LX_T563, (float)0, (int)LX_T564);
  float LX_T566 = native_sqrt(LX_T565);
  X_T566[gout_idx] = LX_T566;
}
