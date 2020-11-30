; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck %s
;
; Source:
;   enum AA { VAL = 100 };
;   typedef int (*func_t)(void);
;   struct s2 { int a[10]; };
;   int test() {
;     func_t f;
;     struct s2 s;
;     enum AA a;
;     return __builtin_preserve_type_info(f, 1) +
;            __builtin_preserve_type_info(s, 1) +
;            __builtin_preserve_type_info(a, 1);
;   }
; Compiler flag to generate IR:
;   clang -target bpf -S -O2 -g -emit-llvm -Xclang -disable-llvm-passes t1.c

target triple = "bpf"

; Function Attrs: nounwind readnone
define dso_local i32 @test() local_unnamed_addr #0 !dbg !17 {
entry:
  call void @llvm.dbg.declare(metadata [10 x i32]* undef, metadata !20, metadata !DIExpression()), !dbg !28
  call void @llvm.dbg.declare(metadata i32 ()** undef, metadata !19, metadata !DIExpression()), !dbg !29
  call void @llvm.dbg.declare(metadata i32* undef, metadata !27, metadata !DIExpression()), !dbg !30
  %0 = tail call i32 @llvm.bpf.preserve.type.info(i32 0, i64 1), !dbg !31, !llvm.preserve.access.index !8
  %1 = tail call i32 @llvm.bpf.preserve.type.info(i32 1, i64 1), !dbg !32, !llvm.preserve.access.index !21
  %add = add i32 %1, %0, !dbg !33
  %2 = tail call i32 @llvm.bpf.preserve.type.info(i32 2, i64 1), !dbg !34, !llvm.preserve.access.index !3
  %add1 = add i32 %add, %2, !dbg !35
  ret i32 %add1, !dbg !36
}

; CHECK:             r{{[0-9]+}} = 8
; CHECK:             r{{[0-9]+}} = 40
; CHECK:             r{{[0-9]+}} = 4
; CHECK:             exit

; CHECK:             .long   16                              # BTF_KIND_TYPEDEF(id = 4)
; CHECK:             .long   49                              # BTF_KIND_STRUCT(id = 7)
; CHECK:             .long   74                              # BTF_KIND_ENUM(id = 10)

; CHECK:             .ascii  ".text"                         # string offset=10
; CHECK:             .ascii  "func_t"                        # string offset=16
; CHECK:             .byte   48                              # string offset=23
; CHECK:             .ascii  "s2"                            # string offset=49
; CHECK:             .ascii  "AA"                            # string offset=74

; CHECK:             .long   16                              # FieldReloc
; CHECK-NEXT:        .long   10                              # Field reloc section string offset=10
; CHECK-NEXT:        .long   3
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   4
; CHECK-NEXT:        .long   23
; CHECK-NEXT:        .long   9
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   7
; CHECK-NEXT:        .long   23
; CHECK-NEXT:        .long   9
; CHECK-NEXT:        .long   .Ltmp{{[0-9]+}}
; CHECK-NEXT:        .long   10
; CHECK-NEXT:        .long   23
; CHECK-NEXT:        .long   9

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare i32 @llvm.bpf.preserve.type.info(i32, i64) #2

attributes #0 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14, !15}
!llvm.ident = !{!16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 12.0.0 (https://github.com/llvm/llvm-project.git d8b1394a0f4bbf57c254f69f8d3aa5381a89b5cd)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !7, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t1.c", directory: "/tmp/home/yhs/tmp1")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "AA", file: !1, line: 1, baseType: !4, size: 32, elements: !5)
!4 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!5 = !{!6}
!6 = !DIEnumerator(name: "VAL", value: 100, isUnsigned: true)
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "func_t", file: !1, line: 2, baseType: !9)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{i32 7, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{!"clang version 12.0.0 (https://github.com/llvm/llvm-project.git d8b1394a0f4bbf57c254f69f8d3aa5381a89b5cd)"}
!17 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 4, type: !10, scopeLine: 4, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !18)
!18 = !{!19, !20, !27}
!19 = !DILocalVariable(name: "f", scope: !17, file: !1, line: 5, type: !8)
!20 = !DILocalVariable(name: "s", scope: !17, file: !1, line: 6, type: !21)
!21 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "s2", file: !1, line: 3, size: 320, elements: !22)
!22 = !{!23}
!23 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !21, file: !1, line: 3, baseType: !24, size: 320)
!24 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 320, elements: !25)
!25 = !{!26}
!26 = !DISubrange(count: 10)
!27 = !DILocalVariable(name: "a", scope: !17, file: !1, line: 7, type: !3)
!28 = !DILocation(line: 6, column: 13, scope: !17)
!29 = !DILocation(line: 5, column: 10, scope: !17)
!30 = !DILocation(line: 7, column: 11, scope: !17)
!31 = !DILocation(line: 8, column: 10, scope: !17)
!32 = !DILocation(line: 9, column: 10, scope: !17)
!33 = !DILocation(line: 8, column: 45, scope: !17)
!34 = !DILocation(line: 10, column: 10, scope: !17)
!35 = !DILocation(line: 9, column: 45, scope: !17)
!36 = !DILocation(line: 8, column: 3, scope: !17)
