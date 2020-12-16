#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 8, 8, 448 }
// Out stride: { 28672, 3584, 448, 1 }
// Elementwise input X_T948 shape: fp32(1, 8, 8, 448):(28672, 3584, 448, 1):112 KiB
// Elementwise input X_T952 shape: fp32(448):(1):1.75 KiB
// Elementwise input X_I_332 shape: fp32(448):(1):1.75 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T953 = div(X_T948, X_T952)
// Elementwise op: [[pid(Add, Switch)]] X_T954 = add(X_T953, X_I_332)
// Elementwise op: X_T955 = cmp_lt(X_T954, X_T2)
// Elementwise op: [[pid(Relu)]] X_T956 = cond(X_T955, X_T2, X_T954)
// Tile size: { 1, 4, 1, 448 }
// Contraction output var shape: fp32(1, 8, 8, 448):(28672, 3584, 448, 1):112 KiB
// Computed true ops: 114688
// Computed work groups: 16
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 672
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 512, 8, 1
__kernel void kernel_c56_sdk_326(__global float* restrict  X_T956, __global const float* restrict  X_T948, __global const float* restrict  X_T952, __global const float* restrict  X_I_332)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 4);
  int i4_tid = (tid % 64);
  int i2_tid = ((tid / 64) % 4);
  for (int i4_lid = 0; i4_lid < 7; i4_lid += 1)
  {
    int i4 = ((64 * i4_lid) + i4_tid);
    int gout_idx = (((3584 * (i2_gid + i2_tid)) + (448 * i3_gid)) + i4);
    float LX_T948 = X_T948[gout_idx];
    float LX_T952 = X_T952[i4];
    float LX_I_332 = X_I_332[i4];
    float LX_T953 = (LX_T948 / LX_T952);
    float LX_T954 = (LX_T953 + LX_I_332);
    int LX_T955 = (LX_T954 < 0.0f);
    float LX_T956 = select((float)LX_T954, (float)0.0f, (int)LX_T955);
    X_T956[gout_idx] = LX_T956;
  }
}
