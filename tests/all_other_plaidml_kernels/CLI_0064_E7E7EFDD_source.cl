#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 9472 37 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 74, 74, 256 }
// Out stride: { 1401856, 18944, 256, 1 }
// Elementwise input X_T170 shape: fp32(1, 74, 74, 256):(1401856, 18944, 256, 1):5476 KiB
// Elementwise input X_T174 shape: fp32(256):(1):1 KiB
// Elementwise input X_I_157 shape: fp32(256):(1):1 KiB
// Elementwise op: [[pid(TrueDiv)]] X_T175 = div(X_T170, X_T174)
// Elementwise op: [[pid(Add, Switch)]] X_T176 = add(X_T175, X_I_157)
// Elementwise op: X_T177 = cmp_lt(X_T176, X_T2)
// Elementwise op: [[pid(Relu)]] X_T178 = cond(X_T177, X_T2, X_T176)
// Tile size: { 1, 2, 2, 256 }
// Contraction output var shape: fp32(1, 74, 74, 256):(1401856, 18944, 256, 1):5476 KiB
// Computed true ops: 5607424
// Computed work groups: 1369
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 4096
// Computed mem read: 384
// Computed mem write: 4096
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 9472, 37, 1
__kernel void kernel_c28_sdk_55(__global float* restrict  X_T178, __global const float* restrict  X_T170, __global const float* restrict  X_T174, __global const float* restrict  X_I_157)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(0) * 2);
  int i2_gid = (get_group_id(1) * 2);
  int i4_tid = (tid % 64);
  int i3_tid = ((tid / 64) % 2);
  int i2_tid = ((tid / 128) % 2);
  for (int i4_lid = 0; i4_lid < 4; i4_lid += 1)
  {
    int i4 = ((64 * i4_lid) + i4_tid);
    int gout_idx = (((18944 * (i2_gid + i2_tid)) + (256 * (i3_gid + i3_tid))) + i4);
    float LX_T170 = X_T170[gout_idx];
    float LX_T174 = X_T174[i4];
    float LX_I_157 = X_I_157[i4];
    float LX_T175 = (LX_T170 / LX_T174);
    float LX_T176 = (LX_T175 + LX_I_157);
    int LX_T177 = (LX_T176 < 0.0f);
    float LX_T178 = select((float)LX_T176, (float)0.0f, (int)LX_T177);
    X_T178[gout_idx] = LX_T178;
  }
}
