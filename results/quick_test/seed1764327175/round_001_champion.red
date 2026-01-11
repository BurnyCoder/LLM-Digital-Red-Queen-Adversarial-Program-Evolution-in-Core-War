
;redcode
;name          Echo Orchard
;author        assistant
;strategy      A small "garden" warrior:
;strategy      - Boots into a safer area
;strategy      - Grows a ring of SPL "vines" (process stunners) outward
;strategy      - Trails DAT bombs behind the growth to punish pursuers
;strategy      The SPL ring tries to flood opponents with extra processes;
;strategy      the DAT trail can kill anything that steps on it.

        ORG     start

step    EQU     37              ; growth spacing (co-prime-ish to many cores)
bootd   EQU     400             ; boot distance

; --- boot block: copy core of program to a distant location, then jump there
start   mov.i   {src,   {dst    ; copy 1
        mov.i   {src,   {dst    ; copy 2
        mov.i   {src,   {dst    ; copy 3
        mov.i   {src,   {dst    ; copy 4
        mov.i   {src,   {dst    ; copy 5
        mov.i   {src,   {dst    ; copy 6
        mov.i   {src,   {dst    ; copy 7
        mov.i   {src,   {dst    ; copy 8
        jmp     boot,   0       ; run the booted copy

src     dat.f   #boot+8, #boot  ; predecrement source pointer (A-field used)
dst     dat.f   #boot+8+bootd, #boot+bootd

; --- main (booted) body: SPL-ring grower + DAT trail
boot    add.ab  #step,  ptr     ; advance target pointer
        spl     @ptr,   0       ; "vine": create process at target (stuns/clogs)
        mov.i   bomb,   <ptr    ; trail a DAT bomb just behind the growth
        jmp     boot,   0       ; keep growing

ptr     dat.f   #0,     #0      ; pointer cell (B-field used by @ and <)
bomb    dat.f   #0,     #0      ; simple lethal bomb

        END
