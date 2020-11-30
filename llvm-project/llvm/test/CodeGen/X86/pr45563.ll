; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -O3 -mtriple=x86_64-linux-generic -mattr=avx < %s | FileCheck %s

; Bug 45563:
; The LowerMLOAD() method AVX masked load branch should
; use the operand vector type rather than the mask type.
; Given, for example:
;   v4f64,ch = masked_load ..
; The select should be:
;   v4f64 = vselect ..
; instead of:
;   v4i64 = vselect ..

define <16 x double> @bug45563(<16 x double>* %addr, <16 x double> %dst, <16 x i64> %e, <16 x i64> %f) {
; CHECK-LABEL: bug45563:
; CHECK:       # %bb.0:
; CHECK-NEXT:    pushq %rbp
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    .cfi_offset %rbp, -16
; CHECK-NEXT:    movq %rsp, %rbp
; CHECK-NEXT:    .cfi_def_cfa_register %rbp
; CHECK-NEXT:    andq $-32, %rsp
; CHECK-NEXT:    subq $32, %rsp
; CHECK-NEXT:    vextractf128 $1, %ymm7, %xmm8
; CHECK-NEXT:    vmovdqa 112(%rbp), %xmm9
; CHECK-NEXT:    vmovdqa 128(%rbp), %xmm10
; CHECK-NEXT:    vpcmpgtq %xmm8, %xmm10, %xmm8
; CHECK-NEXT:    vpcmpgtq %xmm7, %xmm9, %xmm7
; CHECK-NEXT:    vinsertf128 $1, %xmm8, %ymm7, %ymm8
; CHECK-NEXT:    vextractf128 $1, %ymm6, %xmm10
; CHECK-NEXT:    vmovdqa 80(%rbp), %xmm9
; CHECK-NEXT:    vmovdqa 96(%rbp), %xmm7
; CHECK-NEXT:    vpcmpgtq %xmm10, %xmm7, %xmm7
; CHECK-NEXT:    vpcmpgtq %xmm6, %xmm9, %xmm6
; CHECK-NEXT:    vinsertf128 $1, %xmm7, %ymm6, %ymm10
; CHECK-NEXT:    vextractf128 $1, %ymm5, %xmm7
; CHECK-NEXT:    vmovdqa 48(%rbp), %xmm9
; CHECK-NEXT:    vmovdqa 64(%rbp), %xmm6
; CHECK-NEXT:    vpcmpgtq %xmm7, %xmm6, %xmm6
; CHECK-NEXT:    vpcmpgtq %xmm5, %xmm9, %xmm5
; CHECK-NEXT:    vinsertf128 $1, %xmm6, %ymm5, %ymm5
; CHECK-NEXT:    vextractf128 $1, %ymm4, %xmm6
; CHECK-NEXT:    vmovdqa 16(%rbp), %xmm9
; CHECK-NEXT:    vmovdqa 32(%rbp), %xmm7
; CHECK-NEXT:    vpcmpgtq %xmm6, %xmm7, %xmm6
; CHECK-NEXT:    vpcmpgtq %xmm4, %xmm9, %xmm4
; CHECK-NEXT:    vinsertf128 $1, %xmm6, %ymm4, %ymm4
; CHECK-NEXT:    vmaskmovpd (%rdi), %ymm4, %ymm6
; CHECK-NEXT:    vblendvpd %ymm4, %ymm6, %ymm0, %ymm0
; CHECK-NEXT:    vmaskmovpd 32(%rdi), %ymm5, %ymm4
; CHECK-NEXT:    vblendvpd %ymm5, %ymm4, %ymm1, %ymm1
; CHECK-NEXT:    vmaskmovpd 64(%rdi), %ymm10, %ymm4
; CHECK-NEXT:    vblendvpd %ymm10, %ymm4, %ymm2, %ymm2
; CHECK-NEXT:    vmaskmovpd 96(%rdi), %ymm8, %ymm4
; CHECK-NEXT:    vblendvpd %ymm8, %ymm4, %ymm3, %ymm3
; CHECK-NEXT:    movq %rbp, %rsp
; CHECK-NEXT:    popq %rbp
; CHECK-NEXT:    .cfi_def_cfa %rsp, 8
; CHECK-NEXT:    retq
  %mask = icmp slt <16 x i64> %e, %f
  %res = call <16 x double> @llvm.masked.load.v16f64.p0v16f64(<16 x double>* %addr, i32 4, <16 x i1>%mask, <16 x double> %dst)
  ret <16 x double> %res
}

declare <16 x double> @llvm.masked.load.v16f64.p0v16f64(<16 x double>* %addr, i32 %align, <16 x i1> %mask, <16 x double> %dst)
