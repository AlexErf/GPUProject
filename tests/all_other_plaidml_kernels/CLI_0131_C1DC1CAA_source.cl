#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 7168 1 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 28, 28, 22 }
// Out stride: { 17248, 616, 22, 1 }
// Elementwise input X_T412 shape: fp32(1, 28, 28, 22):(17248, 616, 22, 1):67.375 KiB
// Elementwise input X_T416 shape: fp32(22):(1):88 bytes
// Elementwise input X_I_152 shape: fp32(22):(1):88 bytes
// Elementwise input X_T392 shape: fp32(1, 28, 28, 22):(17248, 616, 22, 1):67.375 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T417 = div(X_T412, X_T416)
// Elementwise op: [[pid(Add, Switch)]] X_T418 = add(X_T417, X_I_152)
// Elementwise op: [[pid(Add)]] X_T419 = add(X_T392, X_T418)
// Elementwise op: X_T429 = cmp_lt(X_T419, X_T1)
// Elementwise op: [[pid(Relu)]] X_T430 = cond(X_T429, X_T1, X_T419)
// Tile size: { 1, 28, 1, 22 }
// Contraction output var shape: fp32(1, 28, 28, 22):(17248, 616, 22, 1):67.375 KiB
// Computed true ops: 86240
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 448
// Computed mem write: 7168
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 7168, 1, 1
__kernel void kernel_c42_sdk_142(__global float* restrict  X_T419, __global float* restrict  X_T430, __global const float* restrict  X_T412, __global const float* restrict  X_T416, __global const float* restrict  X_I_152, __global const float* restrict  X_T392)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(0);
  int i4_tid = (tid % 32);
  int i2_tid = ((tid / 32) % 8);
  int i4_cond = (i4_tid < 22);
  if (i4_cond)
  {
    for (int i2_lid = 0; i2_lid < 4; i2_lid += 1)
    {
      int i2_cond = ((i2_lid < 3) || (i2_tid < 4));
      if (i2_cond)
      {
        int i2 = ((8 * i2_lid) + i2_tid);
        int gout_idx = (((616 * i2) + (22 * i3_gid)) + i4_tid);
        float LX_T412 = X_T412[gout_idx];
        float LX_T416 = X_T416[i4_tid];
        float LX_I_152 = X_I_152[i4_tid];
        float LX_T392 = X_T392[gout_idx];
        float LX_T417 = (LX_T412 / LX_T416);
        float LX_T418 = (LX_T417 + LX_I_152);
        float LX_T419 = (LX_T392 + LX_T418);
        int LX_T429 = (LX_T419 < 0.0f);
        float LX_T430 = select((float)LX_T419, (float)0.0f, (int)LX_T429);
        X_T419[gout_idx] = LX_T419;
        X_T430[gout_idx] = LX_T430;
      }
    }
  }
}
