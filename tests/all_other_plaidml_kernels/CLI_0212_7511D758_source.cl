#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 3584 14 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 160 }
// Out stride: { 501760, 8960, 160, 1 }
// Elementwise input X_T167 shape: fp32(1, 56, 56, 160):(501760, 8960, 160, 1):1960 KiB
// Elementwise input X_T171 shape: fp32(160):(1):640 bytes
// Elementwise input X_I_58 shape: fp32(160):(1):640 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T172 = div(X_T167, X_T171)
// Elementwise op: [[pid(Add, Switch)]] X_T173 = add(X_T172, X_I_58)
// Elementwise op: X_T174 = cmp_lt(X_T173, X_T2)
// Elementwise op: [[pid(Relu)]] X_T175 = cond(X_T174, X_T2, X_T173)
// Tile size: { 1, 4, 4, 160 }
// Contraction output var shape: fp32(1, 56, 56, 160):(501760, 8960, 160, 1):1960 KiB
// Computed true ops: 2007040
// Computed work groups: 196
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 10240
// Computed mem read: 960
// Computed mem write: 10240
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 3584, 14, 1
__kernel void kernel_c124_sdk_35(__global float* restrict  X_T175, __global const float* restrict  X_T167, __global const float* restrict  X_T171, __global const float* restrict  X_I_58)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 4);
  int i2_gid = (get_group_id(1) * 4);
  int i4_tid = (tid % 32);
  int i3_tid = ((tid / 32) % 4);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 5; i4_lid += 1)
  {
    int i4 = ((32 * i4_lid) + i4_tid);
    for (int i2_lid = 0; i2_lid < 2; i2_lid += 1)
    {
      int i2 = ((2 * i2_lid) + i2_tid);
      int gout_idx = (((8960 * (i2_gid + i2)) + (160 * (i3_gid + i3_tid))) + i4);
      float LX_T167 = X_T167[gout_idx];
      float LX_T171 = X_T171[i4];
      float LX_I_58 = X_I_58[i4];
      float LX_T172 = (LX_T167 / LX_T171);
      float LX_T173 = (LX_T172 + LX_I_58);
      int LX_T174 = (LX_T173 < 0.0f);
      float LX_T175 = select((float)LX_T173, (float)0.0f, (int)LX_T174);
      X_T175[gout_idx] = LX_T175;
    }
  }
}
