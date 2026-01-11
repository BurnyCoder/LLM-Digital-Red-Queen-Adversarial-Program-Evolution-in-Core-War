
;redcode-94
;name          Moth to a Flame
;author        assistant
;strategy      A compact "lure-and-scorch" scanner:
;strategy      - A small probe advances through core and checks for non-empty cells.
;strategy      - When it sees something that isn't the default DAT 0,0,
;strategy        it drops a tiny SPL/DAT stun/kill pair around that area.
;strategy      - Continues stepping to find more targets.

        ORG     start

step    EQU     29              ; relatively prime-ish step to cover core well
gap     EQU     6               ; spacing between bomb components

probe   DAT.F   #0,     #0      ; will be used as a moving pointer (B-field)

bomb1   SPL.B   #0,     #0      ; process-split "stun" (ties up enemy if executed)
bomb2   DAT.F   #0,     #0      ; hard kill if executed

start   ADD.AB  #step,  probe   ; probe.B += step
scan    SNE.I   $0,     @probe  ; compare current instruction to contents at probe
        JMP.A   bump,   $0      ; if equal (likely empty-ish), keep moving
hit     MOV.I   bomb1,  @probe  ; drop SPL bomb at target
        MOV.I   bomb2,  @probe+gap ; drop DAT a few cells away to catch movement
bump    JMP.A   start,  $0      ; loop forever

        END
