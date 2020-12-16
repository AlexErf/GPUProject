#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1024 8 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 8, 8, 1536 }
// Out stride: { 98304, 12288, 1536, 1 }
// Elementwise input X_T2494 shape: fp32(1, 8, 8, 1536):(98304, 12288, 1536, 1):384 KiB
// Elementwise input X_T2498 shape: fp32(1536):(1):6 KiB
// Elementwise input X_I_2 shape: fp32(1536):(1):6 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T2499 = div(X_T2494, X_T2498)
// Elementwise op: [[pid(Add, Switch)]] X_T2500 = add(X_T2499, X_I_2)
// Elementwise op: X_T2501 = cmp_lt(X_T2500, X_T2)
// Elementwise op: [[pid(Relu)]] X_T2502 = cond(X_T2501, X_T2, X_T2500)
// Tile size: { 1, 2, 1, 1536 }
// Contraction output var shape: fp32(1, 8, 8, 1536):(98304, 12288, 1536, 1):384 KiB
// Computed true ops: 393216
// Computed work groups: 32
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 12288
// Computed mem read: 1152
// Computed mem write: 12288
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1024, 8, 1
__kernel void kernel_c51_sdk_817(__global float* restrict  X_T2502, __global const float* restrict  X_T2494, __global const float* restrict  X_T2498, __global const float* restrict  X_I_2)
{
  int tid = get_local_id(0);
  int i3_gid = get_group_id(1);
  int i2_gid = (get_group_id(0) * 2);
  int i4_tid = (tid % 128);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 12; i4_lid += 1)
  {
    int i4 = ((128 * i4_lid) + i4_tid);
    int gout_idx = (((12288 * (i2_gid + i2_tid)) + (1536 * i3_gid)) + i4);
    float LX_T2494 = X_T2494[gout_idx];
    float LX_T2498 = X_T2498[i4];
    float LX_I_2 = X_I_2[i4];
    float LX_T2499 = (LX_T2494 / LX_T2498);
    float LX_T2500 = (LX_T2499 + LX_I_2);
    int LX_T2501 = (LX_T2500 < 0.0f);
    float LX_T2502 = select((float)LX_T2500, (float)0.0f, (int)LX_T2501);
    X_T2502[gout_idx] = LX_T2502;
  }
}
