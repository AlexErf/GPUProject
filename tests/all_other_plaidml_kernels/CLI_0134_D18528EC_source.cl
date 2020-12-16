#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1536 28 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 192 }
// Out stride: { 602112, 10752, 192, 1 }
// Elementwise input X_T164 shape: fp32(1, 56, 56, 192):(602112, 10752, 192, 1):2352 KiB
// Elementwise input X_T168 shape: fp32(192):(1):768 bytes
// Elementwise input X_I_68 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T169 = div(X_T164, X_T168)
// Elementwise op: [[pid(Add, Switch)]] X_T170 = add(X_T169, X_I_68)
// Elementwise op: X_T171 = cmp_lt(X_T170, X_T2)
// Elementwise op: [[pid(Relu)]] X_T172 = cond(X_T171, X_T2, X_T170)
// Tile size: { 1, 56, 2, 32 }
// Contraction output var shape: fp32(1, 56, 56, 192):(602112, 10752, 192, 1):2352 KiB
// Computed true ops: 2408448
// Computed work groups: 168
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 14336
// Computed mem read: 1344
// Computed mem write: 14336
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1536, 28, 1
__kernel void kernel_c68_sdk_44(__global float* restrict  X_T172, __global const float* restrict  X_T164, __global const float* restrict  X_T168, __global const float* restrict  X_I_68)
{
  int tid = get_local_id(0);
  int i4_gid = (get_group_id(0) * 32);
  int i3_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 2);
  int i2_tid = ((tid / 64) % 4);
  for (int i2_lid = 0; i2_lid < 14; i2_lid += 1)
  {
    int i2 = ((4 * i2_lid) + i2_tid);
    int gout_idx = (((10752 * i2) + (192 * (i3_gid + i3_tid))) + (i4_gid + i4_tid));
    float LX_T164 = X_T164[gout_idx];
    float LX_T168 = X_T168[(i4_gid + i4_tid)];
    float LX_I_68 = X_I_68[(i4_gid + i4_tid)];
    float LX_T169 = (LX_T164 / LX_T168);
    float LX_T170 = (LX_T169 + LX_I_68);
    int LX_T171 = (LX_T170 < 0.0f);
    float LX_T172 = select((float)LX_T170, (float)0.0f, (int)LX_T171);
    X_T172[gout_idx] = LX_T172;
  }
}
