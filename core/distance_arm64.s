//go:build arm64

#include "textflag.h"

// The Go arm64 assembler has no vector FADD/FSUB/FADDP, so those are emitted
// as raw WORD encodings.

// func dotSIMD(a, b *float32, n int) float32
TEXT ·dotSIMD(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16
	VEOR V20.B16, V20.B16, V20.B16
	VEOR V21.B16, V21.B16, V21.B16
	VEOR V22.B16, V22.B16, V22.B16
	VEOR V23.B16, V23.B16, V23.B16
	VEOR V24.B16, V24.B16, V24.B16

	CMP $32, R2
	BLT dot_loop4

dot_loop32:
	VLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1.P 64(R0), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P 64(R1), [V8.S4, V9.S4, V10.S4, V11.S4]
	VLD1.P 64(R1), [V12.S4, V13.S4, V14.S4, V15.S4]
	VFMLA V0.S4, V8.S4, V16.S4
	VFMLA V1.S4, V9.S4, V17.S4
	VFMLA V2.S4, V10.S4, V18.S4
	VFMLA V3.S4, V11.S4, V19.S4
	VFMLA V4.S4, V12.S4, V20.S4
	VFMLA V5.S4, V13.S4, V21.S4
	VFMLA V6.S4, V14.S4, V22.S4
	VFMLA V7.S4, V15.S4, V23.S4
	SUB $32, R2
	CMP $32, R2
	BGE dot_loop32

dot_loop4:
	CMP $4, R2
	BLT dot_tail
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R1), [V8.S4]
	VFMLA V0.S4, V8.S4, V16.S4
	SUB $4, R2
	B dot_loop4

dot_tail:
	CBZ R2, dot_reduce
	FMOVS.P 4(R0), F0
	FMOVS.P 4(R1), F1
	FMULS F0, F1, F0
	FADDS F0, F24, F24
	SUB $1, R2
	B dot_tail

dot_reduce:
	WORD $0x4e31d610 // fadd v16.4s, v16.4s, v17.4s
	WORD $0x4e33d652 // fadd v18.4s, v18.4s, v19.4s
	WORD $0x4e35d694 // fadd v20.4s, v20.4s, v21.4s
	WORD $0x4e37d6d6 // fadd v22.4s, v22.4s, v23.4s
	WORD $0x4e32d610 // fadd v16.4s, v16.4s, v18.4s
	WORD $0x4e36d694 // fadd v20.4s, v20.4s, v22.4s
	WORD $0x4e34d610 // fadd v16.4s, v16.4s, v20.4s
	WORD $0x6e30d610 // faddp v16.4s, v16.4s, v16.4s
	WORD $0x7e30da10 // faddp s16, v16.2s
	FADDS F24, F16, F16
	FMOVS F16, ret+24(FP)
	RET

// func l2SqSIMD(a, b *float32, n int) float32
TEXT ·l2SqSIMD(SB), NOSPLIT, $0-28
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2
	VEOR V16.B16, V16.B16, V16.B16
	VEOR V17.B16, V17.B16, V17.B16
	VEOR V18.B16, V18.B16, V18.B16
	VEOR V19.B16, V19.B16, V19.B16
	VEOR V20.B16, V20.B16, V20.B16
	VEOR V21.B16, V21.B16, V21.B16
	VEOR V22.B16, V22.B16, V22.B16
	VEOR V23.B16, V23.B16, V23.B16
	VEOR V24.B16, V24.B16, V24.B16

	CMP $32, R2
	BLT l2_loop4

l2_loop32:
	VLD1.P 64(R0), [V0.S4, V1.S4, V2.S4, V3.S4]
	VLD1.P 64(R0), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD1.P 64(R1), [V8.S4, V9.S4, V10.S4, V11.S4]
	VLD1.P 64(R1), [V12.S4, V13.S4, V14.S4, V15.S4]
	WORD $0x4ea8d408 // fsub v8.4s, v0.4s, v8.4s
	WORD $0x4ea9d429 // fsub v9.4s, v1.4s, v9.4s
	WORD $0x4eaad44a // fsub v10.4s, v2.4s, v10.4s
	WORD $0x4eabd46b // fsub v11.4s, v3.4s, v11.4s
	WORD $0x4eacd48c // fsub v12.4s, v4.4s, v12.4s
	WORD $0x4eadd4ad // fsub v13.4s, v5.4s, v13.4s
	WORD $0x4eaed4ce // fsub v14.4s, v6.4s, v14.4s
	WORD $0x4eafd4ef // fsub v15.4s, v7.4s, v15.4s
	VFMLA V8.S4, V8.S4, V16.S4
	VFMLA V9.S4, V9.S4, V17.S4
	VFMLA V10.S4, V10.S4, V18.S4
	VFMLA V11.S4, V11.S4, V19.S4
	VFMLA V12.S4, V12.S4, V20.S4
	VFMLA V13.S4, V13.S4, V21.S4
	VFMLA V14.S4, V14.S4, V22.S4
	VFMLA V15.S4, V15.S4, V23.S4
	SUB $32, R2
	CMP $32, R2
	BGE l2_loop32

l2_loop4:
	CMP $4, R2
	BLT l2_tail
	VLD1.P 16(R0), [V0.S4]
	VLD1.P 16(R1), [V8.S4]
	WORD $0x4ea8d408 // fsub v8.4s, v0.4s, v8.4s
	VFMLA V8.S4, V8.S4, V16.S4
	SUB $4, R2
	B l2_loop4

l2_tail:
	CBZ R2, l2_reduce
	FMOVS.P 4(R0), F0
	FMOVS.P 4(R1), F1
	FSUBS F1, F0, F0
	FMULS F0, F0, F0
	FADDS F0, F24, F24
	SUB $1, R2
	B l2_tail

l2_reduce:
	WORD $0x4e31d610 // fadd v16.4s, v16.4s, v17.4s
	WORD $0x4e33d652 // fadd v18.4s, v18.4s, v19.4s
	WORD $0x4e35d694 // fadd v20.4s, v20.4s, v21.4s
	WORD $0x4e37d6d6 // fadd v22.4s, v22.4s, v23.4s
	WORD $0x4e32d610 // fadd v16.4s, v16.4s, v18.4s
	WORD $0x4e36d694 // fadd v20.4s, v20.4s, v22.4s
	WORD $0x4e34d610 // fadd v16.4s, v16.4s, v20.4s
	WORD $0x6e30d610 // faddp v16.4s, v16.4s, v16.4s
	WORD $0x7e30da10 // faddp s16, v16.2s
	FADDS F24, F16, F16
	FMOVS F16, ret+24(FP)
	RET
