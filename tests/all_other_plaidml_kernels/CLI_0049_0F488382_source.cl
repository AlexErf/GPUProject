#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 512 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 512 }
// Out stride: { 1 }
// Elementwise input X_I_266 shape: fp32(512):(1):2 KiB
// Elementwise op: [[pid(Add)]] X_T198 = add(X_T37, X_I_266)
// Elementwise op: X_T199 = cmp_lt(X_T198, X_T3)
// Elementwise op: [[pid(Sqrt)]] X_T200 = cond(X_T199, X_T3, X_T198)
// Elementwise op: [[pid(Sqrt)]] X_T201 = sqrt(X_T200)
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
__kernel void kernel_c29_sdk_43(__global float* restrict  X_T201, __global const float* restrict  X_I_266)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int gout_idx = (i1_gid + i1_tid);
  float LX_I_266 = X_I_266[gout_idx];
  float LX_T198 = (0.0010000000474974513f + LX_I_266);
  int LX_T199 = (LX_T198 < (float)0);
  float LX_T200 = select((float)LX_T198, (float)0, (int)LX_T199);
  float LX_T201 = native_sqrt(LX_T200);
  X_T201[gout_idx] = LX_T201;
}
