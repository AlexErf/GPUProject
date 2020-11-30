; NOTE: Assertions have been autogenerated by utils/update_analyze_test_checks.py
; RUN: opt < %s -S -mtriple=thumbv8.1m.main-none-eabi -mattr=+mve.fp -cost-model -analyze | FileCheck %s

; Note that these instructions like this (not in a look that could be tail
; predicated) should not really be free. We currently assume that all active
; lane masks are free.

define void @v4i32(i32 %index, i32 %TC) {
; CHECK-LABEL: 'v4i32'
; CHECK-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %index, i32 %TC)
; CHECK-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
  %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %index, i32 %TC)
  ret void
}

define void @v8i16(i32 %index, i32 %TC) {
; CHECK-LABEL: 'v8i16'
; CHECK-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %active.lane.mask = call <8 x i1> @llvm.get.active.lane.mask.v8i1.i32(i32 %index, i32 %TC)
; CHECK-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
  %active.lane.mask = call <8 x i1> @llvm.get.active.lane.mask.v8i1.i32(i32 %index, i32 %TC)
  ret void
}

define void @v16i8(i32 %index, i32 %TC) {
; CHECK-LABEL: 'v16i8'
; CHECK-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %active.lane.mask = call <16 x i1> @llvm.get.active.lane.mask.v16i1.i32(i32 %index, i32 %TC)
; CHECK-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: ret void
;
  %active.lane.mask = call <16 x i1> @llvm.get.active.lane.mask.v16i1.i32(i32 %index, i32 %TC)
  ret void
}

declare <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32, i32)
declare <8 x i1> @llvm.get.active.lane.mask.v8i1.i32(i32, i32)
declare <16 x i1> @llvm.get.active.lane.mask.v16i1.i32(i32, i32)
