
    /*
     * Calling convention:
     * Integer and pointer arguments: rdi, rsi, rdx, rcx, r8, r9
     * Floating point: xmm0-7
     * The rest: stack
     */

    .global sdirprod_stride_all_1_aaa
    .type sdirprod_stride_all_1_aaa,@function
    .arch .avx2
    /**
     * Direct product kernel for when all strides are 1 and all pointers are aligned to 32 bits.
     */
sdirprod_stride_all_1_aaa:
    .cfi_startproc
    pushq   %rbp
    movq    %rsp, %rbp
    subq    $128, %rsp

    // Loop through any unaligned data at the beginning.
    // First, calculate the number of unaligned data points.
    movq    %rsi, %rax
    andq    $31, %rax
    shrq    $2, %rax

    // If it is greater than the number of elements, then this becomes the number of elements.
    cmpq    %rdi, %rax
    cmovgq  %rdi, %rax

    // Subtract this number from rdi to get the bulk of the aligned elements.
    subq    %rax, %rdi

    // Set up our increment. cl = 0. Use 8 bit registers for SPEED!
    pushq   %rbx
    movq    %rcx, %rbx
    xorb    %cl, %cl

    // While cl is less than al, then direct product.
.begin_unvector_loop_1_aaa:
    cmpb    %al, %cl
    jge     .end_unvector_loop_1_aaa

    // Load values.
    movss   (%rsi), %xmm2
    movss   (%rdx), %xmm3
    movss   (%rbx), %xmm4
    // Multiply by alpha and beta.
    mulss   %xmm0, %xmm2
    mulss   %xmm1, %xmm4

    // Multiply and add. Backwards because AT&T, so xmm0 * xmm1 + xmm2 = xmm2.
    vfmadd231ss %xmm2, %xmm3, %xmm4

    // Save
    movss   %xmm4, (%rbx)

    // Increment.
    incb    %cl
    leaq    4(%rsi), %rsi
    leaq    4(%rdx), %rdx
    leaq    4(%rbx), %rbx
    jmp     .begin_unvector_loop_1_aaa
.end_unvector_loop_1_aaa:

    // Now, we are aligned. Subtract the number of elements we needed to use from rsi.
    subq    %rax, %rdi

    // Now, calculate the excess elements after the vector stuff.
    movq    %rdi, %rcx
    andq    $7, %rcx
    movq    %rdi, %rax
    subq    %rcx, %rax

    // Now the number of excess elements are given in rcx and the number of aligned elements are given in rax.
    // We can now shift rax to give the number of vector operations we need.
    shrq    $3, %rax

    // Save the number of excess.
    pushq   %rcx

    // Broadcast the prefactors.
    movups  %xmm0, %xmm2
    movups  %xmm1, %xmm3
    vbroadcastss    %xmm2, %ymm0
    vbroadcastss    %xmm3, %ymm1

    // Now, loop.
    xorq    %rcx, %rcx
.begin_vector_loop_1_aaa:
    cmpq    %rax, %rcx
    jge     .end_vector_loop_1_aaa

    // Load the data.
    vmovups     (%rsi), %ymm2
    vmovups     (%rdx), %ymm3
    vmovups     (%rbx), %ymm4

    // Scale the data.
    vmulps  %ymm0, %ymm2, %ymm2
    vmulps  %ymm1, %ymm4, %ymm4

    // Multiply and add.
    vfmadd231ps     %ymm2, %ymm3, %ymm4

    // Store the data.
    vmovups     %ymm4, (%rbx)

    // Increment rcx.
    incq    %rcx
    leaq    32(%rsi), %rsi
    leaq    32(%rdx), %rdx
    leaq    32(%rbx), %rbx

    jmp     .begin_vector_loop_1_aaa
.end_vector_loop_1_aaa:

    // Now handle the rest of the elements.
    popq    %rax
    xorb    %cl, %cl

.begin_unvector_loop_1_aaa_2:
    cmpb    %al, %cl
    jge     .end_unvector_loop_1_aaa_2

    // Load values.
    movss   (%rsi), %xmm2
    movss   (%rdx), %xmm3
    movss   (%rbx), %xmm4

    // Scale by the prefactors.
    mulss   %xmm0, %xmm2
    mulss   %xmm1, %xmm4

    // Multiply and add. Backwards because AT&T, so xmm2 * xmm3 + xmm4 = xmm4.
    vfmadd231ss %xmm2, %xmm3, %xmm4

    // Save
    movss   %xmm4, (%rbx)

    // Increment.
    incb    %cl
    leaq    4(%rsi), %rsi
    leaq    4(%rdx), %rdx
    leaq    4(%rbx), %rbx
    jmp     .begin_unvector_loop_1_aaa_2
.end_unvector_loop_1_aaa_2:

    // Pop the saved rbx.
    popq    %rbx

    // Undo the stack frame.
    addq    $128, %rsp
    popq    %rbp
    retq


    .cfi_endproc

    .global sdirprod_stride_all_1
    .arch .avx2
    .type sdirprod_stride_all_1, @function
sdirprod_stride_all_1:
    .cfi_startproc
    pushq   %rbp
    movq    %rsp, %rbp
    subq    $128, %rsp
    


    .cfi_endproc

    .global sdirprod
    .arch .avx2
    .type sdirprod, @function
sdirprod:
    // Set up the stack.
    .cfi_startproc
    pushq   %rbp
    movq    %rsp, %rbp
    subq    $128, %rsp

    pushq   %rdi
    pushq   %rsi
    pushq   %rdx
    pushq   %rcx
    pushq   %r8
    pushq   %r9
    movss   %xmm0, -16(%rbp)
    movss   %xmm1, -32(%rbp)

    // The arguments should be passed as follows:
    // rdi: n
    // rsi: x
    // rdx: incx
    // rcx: y
    // r8: incy
    // r9: z
    // (%rbp): old base
    // 8(%rbp): return address
    // 16(%rbp): incz

    // We need to pull incz into the current stack frame.
    movq    16(%rbp), %rax

    // Now, check to see if the strides are all 1.
    cmpq    $1, %rdx
    jne     .s_handle_non_unit
    cmpq    $1, %r8
    jne     .s_handle_non_unit
    cmpq    $1, %rax
    jne     .s_handle_non_unit

    // Now, check to see if the pointers are all aligned.

    testq   $3, %rsi
    jnz     .s_handle_unaligned
    testq   $3, %rcx
    jnz     .s_handle_unaligned
    testq   $3, %r9
    jnz     .s_handle_unaligned

    // Set up for a call to the inner kernel.
    pushq   %rdx
    pushq   %r8
    
    movq    %rcx, %rdx
    movq    %r9, %rcx

    call    sdirprod_stride_all_1_aaa

    popq    %rdx
    popq    %r8

    jmp     .s_end_of_func


.s_handle_unaligned:
.s_handle_non_unit:
.s_end_of_func:

    movss   -32(%rbp), %xmm1
    movss   -16(%rbp), %xmm0
    popq    %r9
    popq    %r8
    popq    %rcx
    popq    %rdx
    popq    %rsi
    popq    %rdi

    addq    $128, %rsp
    popq    %rbp
    retq

    .cfi_endproc