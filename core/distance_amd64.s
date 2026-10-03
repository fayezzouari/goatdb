// AVX2 distance kernels for amd64.
// All functions use the stack-based (ABI0) calling convention.
// Arguments layout matches the Go declarations in distance_amd64.go.

#include "textflag.h"

// func dotSIMD(a, b *float32, n int) float32
// Computes sum(a[i]*b[i]) using AVX2 FMA, unrolled 2x (16 floats/iter).
TEXT ·dotSIMD(SB), NOSPLIT, $0-28
    MOVQ a+0(FP), SI
    MOVQ b+8(FP), DI
    MOVQ n+16(FP), CX

    VXORPS Y0, Y0, Y0       // acc0 = 0
    VXORPS Y1, Y1, Y1       // acc1 = 0

    MOVQ CX, AX
    SHRQ $4, AX             // AX = n/16
    JZ   tail8

loop16:
    VMOVUPS 0(SI), Y2
    VMOVUPS 32(SI), Y3
    VFMADD231PS 0(DI), Y2, Y0
    VFMADD231PS 32(DI), Y3, Y1
    ADDQ $64, SI
    ADDQ $64, DI
    DECQ AX
    JNZ  loop16

tail8:
    TESTQ $8, CX
    JZ    hsum
    VMOVUPS 0(SI), Y2
    VFMADD231PS 0(DI), Y2, Y0
    ADDQ $32, SI
    ADDQ $32, DI

hsum:
    VADDPS Y1, Y0, Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    ANDQ $7, CX
    JZ   done_dot

scalar_dot:
    MOVSS  0(SI), X2
    MULSS  0(DI), X2
    ADDSS  X2, X0
    ADDQ   $4, SI
    ADDQ   $4, DI
    DECQ   CX
    JNZ    scalar_dot

done_dot:
    VZEROUPPER
    MOVSS X0, ret+24(FP)
    RET

// func l2SqSIMD(a, b *float32, n int) float32
// Computes sum((a[i]-b[i])^2) using AVX2, unrolled 2x.
TEXT ·l2SqSIMD(SB), NOSPLIT, $0-28
    MOVQ a+0(FP), SI
    MOVQ b+8(FP), DI
    MOVQ n+16(FP), CX

    VXORPS Y0, Y0, Y0
    VXORPS Y1, Y1, Y1

    MOVQ CX, AX
    SHRQ $4, AX
    JZ   tail8_l2

loop16_l2:
    VMOVUPS 0(SI), Y2
    VMOVUPS 32(SI), Y3
    VMOVUPS 0(DI), Y4
    VMOVUPS 32(DI), Y5
    VSUBPS Y4, Y2, Y2
    VSUBPS Y5, Y3, Y3
    VFMADD231PS Y2, Y2, Y0
    VFMADD231PS Y3, Y3, Y1
    ADDQ $64, SI
    ADDQ $64, DI
    DECQ AX
    JNZ  loop16_l2

tail8_l2:
    TESTQ $8, CX
    JZ    hsum_l2
    VMOVUPS 0(SI), Y2
    VMOVUPS 0(DI), Y4
    VSUBPS Y4, Y2, Y2
    VFMADD231PS Y2, Y2, Y0
    ADDQ $32, SI
    ADDQ $32, DI

hsum_l2:
    VADDPS Y1, Y0, Y0
    VEXTRACTF128 $1, Y0, X1
    VADDPS X1, X0, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    ANDQ $7, CX
    JZ   done_l2

scalar_l2:
    MOVSS  0(SI), X2
    SUBSS  0(DI), X2
    MULSS  X2, X2
    ADDSS  X2, X0
    ADDQ   $4, SI
    ADDQ   $4, DI
    DECQ   CX
    JNZ    scalar_l2

done_l2:
    VZEROUPPER
    MOVSS X0, ret+24(FP)
    RET
