
;redcode
;name          Mirage Garden
;author        ChatGPT
;strategy      A small "garden" that quickly grows into multiple processes,
;strategy      lays a ring of decoy instructions around itself, then launches
;strategy      two imps in opposite directions. The decoys try to waste enemy
;strategy      scanners and misdirect simple bombers.

        ORG     start

step    EQU     23          ; relatively prime to many common CORESIZE values
gap     EQU     97          ; spacing for decoy planting

; --- core data / pointers ---
ptr     DAT.F   #0,     #gap        ; B-field is the moving decoy pointer
seed    DAT.F   #step,  #step       ; used to advance ptr

; --- decoy instructions ---
dec1    MOV.I   <ptr,   >ptr        ; looks like a copier; also perturbs ptr cell
dec2    SNE.I   *ptr,   @ptr        ; noisy compare to confuse scanners
dec3    JMP.B   @ptr,   {ptr        ; indirect jump bait

; --- imp bodies (two directions) ---
impf    MOV.I   0,      1           ; forward imp
impb    MOV.I   0,     -1           ; backward imp

; --- bootstrap / main ---
start   SPL.B   1,      0           ; create a second process immediately
grow    SPL.B   plant,  0           ; dedicate one process to planting decoys
        SPL.B   launch, 0           ; another process to launch imps
loop    ADD.F   seed,   ptr         ; advance the decoy pointer (both fields)
        JMP.B   loop,   0           ; keep one process alive in a tight loop

; --- decoy planting process ---
plant   MOV.I   dec1,   @ptr        ; plant decoy 1 at moving target
        ADD.AB  #1,     ptr         ; nudge pointer to change pattern
        MOV.I   dec2,   @ptr        ; plant decoy 2
        ADD.AB  #1,     ptr
        MOV.I   dec3,   @ptr        ; plant decoy 3
        ADD.F   seed,   ptr         ; jump ahead by step+gap mixture
        JMP.B   plant,  0

; --- imp launching process ---
launch  SPL.B   impstart,0          ; start forward imp
        JMP.B   impback, 0          ; current process becomes backward imp

impstart JMP.B  impf,    0          ; enter forward imp loop
impback  JMP.B  impb,    0          ; enter backward imp loop

        END
