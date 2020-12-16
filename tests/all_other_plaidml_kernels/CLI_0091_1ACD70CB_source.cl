#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 56, 56, 11 }
// Out stride: { 34496, 616, 11, 1 }
// Elementwise input X_T190 shape: fp32(1, 56, 56, 11):(34496, 616, 11, 1):134.75 KiB
// Elementwise input X_T194 shape: fp32(11):(1):44 bytes
// Elementwise input X_I_85 shape: fp32(11):(1):44 bytes
// Elementwise input X_T170 shape: fp32(1, 56, 56, 11):(34496, 616, 11, 1):134.75 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T195 = div(X_T190, X_T194)
// Elementwise op: [[pid(Add, Switch)]] X_T196 = add(X_T195, X_I_85)
// Elementwise op: [[pid(Add)]] X_T197 = add(X_T170, X_T196)
// Elementwise op: X_T207 = cmp_lt(X_T197, X_T1)
// Elementwise op: [[pid(Relu)]] X_T208 = cond(X_T207, X_T1, X_T197)
// Tile size: { 1, 56, 2, 11 }
// Contraction output var shape: fp32(1, 56, 56, 11):(34496, 616, 11, 1):134.75 KiB
// Computed true ops: 172480
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 7168
// Computed mem read: 1792
// Computed mem write: 28672
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c42_sdk_56(__global float* restrict  X_T197, __global float* restrict  X_T208, __global const float* restrict  X_T190, __global const float* restrict  X_T194, __global const float* restrict  X_I_85, __global const float* restrict  X_T170)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 16);
  int i3_tid = ((tid / 16) % 2);
  int i2_tid = ((tid / 32) % 8);
  int i4_cond = (i4_tid < 11);
  if (i4_cond)
  {
    for (int i2_lid = 0; i2_lid < 7; i2_lid += 1)
    {
      int i2 = ((8 * i2_lid) + i2_tid);
      int gout_idx = (((616 * i2) + (11 * (i3_gid + i3_tid))) + i4_tid);
      float LX_T190 = X_T190[gout_idx];
      float LX_T194 = X_T194[i4_tid];
      float LX_I_85 = X_I_85[i4_tid];
      float LX_T170 = X_T170[gout_idx];
      float LX_T195 = (LX_T190 / LX_T194);
      float LX_T196 = (LX_T195 + LX_I_85);
      float LX_T197 = (LX_T170 + LX_T196);
      int LX_T207 = (LX_T197 < 0.0f);
      float LX_T208 = select((float)LX_T197, (float)0.0f, (int)LX_T207);
      X_T197[gout_idx] = LX_T197;
      X_T208[gout_idx] = LX_T208;
    }
  }
}
