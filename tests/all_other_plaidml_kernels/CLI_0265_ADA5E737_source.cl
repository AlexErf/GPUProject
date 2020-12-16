#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 448 }
// Out stride: { 351232, 12544, 448, 1 }
// Elementwise input X_T478 shape: fp32(1, 28, 28, 448):(351232, 12544, 448, 1):1372 KiB
// Elementwise input X_T501 shape: fp32(1, 28, 28, 448):(351232, 12544, 448, 1):1372 KiB
// Elementwise input X_I_191 shape: fp32(448):(1):1.75 KiB
// Elementwise input X_I_190 shape: fp32(448):(1):1.75 KiB
// Elementwise op: [[pid(Concatenate)]] X_T502 = add(X_T478, X_T501)
// Elementwise op: [[pid(Sub)]] X_T504 = sub(X_T502, X_I_191)
// Elementwise op: [[pid(Mul)]] X_T505 = mul(X_T504, X_I_190)
// Tile size: { 1, 28, 2, 32 }
// Contraction output var shape: fp32(1, 28, 28, 448):(351232, 12544, 448, 1):1372 KiB
// Computed true ops: 1053696
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 896
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 14, 1
__kernel void kernel_c108_sdk_155(__global float* restrict  X_T502, __global float* restrict  X_T505, __global const float* restrict  X_T478, __global const float* restrict  X_T501, __global const float* restrict  X_I_191, __global const float* restrict  X_I_190)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 32);
  int i3_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 2);
  int i2_tid = ((tid / 64) % 4);
  for (int i2_lid = 0; i2_lid < 7; i2_lid += 1)
  {
    int i2 = ((4 * i2_lid) + i2_tid);
    int gout_idx = (((12544 * i2) + (448 * (i3_gid + i3_tid))) + (i4_gid + i4_tid));
    float LX_T478 = X_T478[gout_idx];
    float LX_T501 = X_T501[gout_idx];
    float LX_I_191 = X_I_191[(i4_gid + i4_tid)];
    float LX_I_190 = X_I_190[(i4_gid + i4_tid)];
    float LX_T502 = (LX_T478 + LX_T501);
    float LX_T504 = (LX_T502 - LX_I_191);
    float LX_T505 = (LX_T504 * LX_I_190);
    X_T502[gout_idx] = LX_T502;
    X_T505[gout_idx] = LX_T505;
  }
}
