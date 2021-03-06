#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 768 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 736 }
// Out stride: { 1 }
// Elementwise input X_I_363 shape: fp32(736):(1):2.875 KiB
// Elementwise op: [[pid(Add)]] X_T951 = add(X_T63, X_I_363)
// Elementwise op: X_T952 = cmp_lt(X_T951, X_T36)
// Elementwise op: [[pid(Sqrt)]] X_T953 = cond(X_T952, X_T36, X_T951)
// Elementwise op: [[pid(Sqrt)]] X_T954 = sqrt(X_T953)
// Tile size: { 256 }
// Contraction output var shape: fp32(736):(1):2.875 KiB
// Computed true ops: 2944
// Computed work groups: 3
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 32
// Computed mem write: 1024
// Computed operations: 256
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 768, 1, 1
__kernel void kernel_c108_sdk_316(__global float* restrict  X_T954, __global const float* restrict  X_I_363)
{
  int tid = get_local_id(0);
  int i1_gid = (get_group_id(0) * 256);
  int i1_tid = (tid % 256);
  int i1_cond = ((i1_gid != 512) || (i1_tid < 224));
  if (i1_cond)
  {
    int gout_idx = (i1_gid + i1_tid);
    float LX_I_363 = X_I_363[gout_idx];
    float LX_T951 = (1.0009999641624745e-5f + LX_I_363);
    int LX_T952 = (LX_T951 < (float)0);
    float LX_T953 = select((float)LX_T951, (float)0, (int)LX_T952);
    float LX_T954 = native_sqrt(LX_T953);
    X_T954[gout_idx] = LX_T954;
  }
}
