#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// gid: 256 1 1
// lid: 256 1 1
// Names: { i1 }
// Ranges: { 192 }
// Out stride: { 1 }
// Elementwise input X_I_71 shape: fp32(192):(1):768 bytes
// Elementwise op: [[pid(Add)]] X_T165 = add(X_T43, X_I_71)
// Elementwise op: X_T166 = cmp_lt(X_T165, X_T20)
// Elementwise op: [[pid(Sqrt)]] X_T167 = cond(X_T166, X_T20, X_T165)
// Elementwise op: [[pid(Sqrt)]] X_T168 = sqrt(X_T167)
// Tile size: { 192 }
// Contraction output var shape: fp32(192):(1):768 bytes
// Computed true ops: 768
// Computed work groups: 1
// Computed inner loops: 1
// Computed shared mem: 0
// Computed out regs: 1024
// Computed mem read: 24
// Computed mem write: 768
// Computed operations: 192
// Computed rollups: 0
// Computed threads used: 256
// lwork = 256, 1, 1
// gwork = 256, 1, 1
__kernel void kernel_c68_sdk_43(__global float* restrict  X_T168, __global const float* restrict  X_I_71)
{
  int tid = get_local_id(0);
  int i1_tid = (tid % 256);
  int i1_cond = (i1_tid < 192);
  if (i1_cond)
  {
    float LX_I_71 = X_I_71[i1_tid];
    float LX_T165 = (1.0009999641624745e-5f + LX_I_71);
    int LX_T166 = (LX_T165 < (float)0);
    float LX_T167 = select((float)LX_T165, (float)0, (int)LX_T166);
    float LX_T168 = native_sqrt(LX_T167);
    X_T168[i1_tid] = LX_T168;
  }
}
