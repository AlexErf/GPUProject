#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 1792 4 1
// lid: 256 1 1
// Names: { i1, i2, i3, i4 }
// Ranges: { 1, 7, 7, 128 }
// Out stride: { 6272, 896, 128, 1 }
// Elementwise input X_T1814 shape: fp32(1, 7, 7, 128):(6272, 896, 128, 1):24.5 KiB
// Elementwise input X_T1818 shape: fp32(128):(1):512 bytes
// Elementwise input X_I_687 shape: fp32(128):(1):512 bytes
// Elementwise op: [[pid(TrueDiv)]] X_T1819 = div(X_T1814, X_T1818)
// Elementwise op: [[pid(Add, Switch)]] X_T1820 = add(X_T1819, X_I_687)
// Elementwise op: X_T1821 = cmp_lt(X_T1820, X_T2)
// Elementwise op: [[pid(Relu)]] X_T1822 = cond(X_T1821, X_T2, X_T1820)
// Tile size: { 1, 1, 2, 128 }
// Contraction output var shape: fp32(1, 7, 7, 128):(6272, 896, 128, 1):24.5 KiB
// Computed true ops: 25088
// Computed work groups: 28
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 96
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 1792, 4, 1
__kernel void kernel_c124_sdk_623(__global float* restrict  X_T1822, __global const float* restrict  X_T1814, __global const float* restrict  X_T1818, __global const float* restrict  X_I_687)
{
  int tid = get_local_id(0);
  int i3_gid = (get_group_id(1) * 2);
  int i2_gid = get_group_id(0);
  int i4_tid = (tid % 128);
  int i3_tid = ((tid / 128) % 2);
  int i3_cond = ((i3_gid != 6) || (i3_tid < 1));
  if (i3_cond)
  {
    int gout_idx = (((896 * i2_gid) + (128 * (i3_gid + i3_tid))) + i4_tid);
    float LX_T1814 = X_T1814[gout_idx];
    float LX_T1818 = X_T1818[i4_tid];
    float LX_I_687 = X_I_687[i4_tid];
    float LX_T1819 = (LX_T1814 / LX_T1818);
    float LX_T1820 = (LX_T1819 + LX_I_687);
    int LX_T1821 = (LX_T1820 < 0.0f);
    float LX_T1822 = select((float)LX_T1820, (float)0.0f, (int)LX_T1821);
    X_T1822[gout_idx] = LX_T1822;
  }
}
