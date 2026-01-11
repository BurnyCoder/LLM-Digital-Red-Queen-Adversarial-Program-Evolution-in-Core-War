
        ORG     start
; three-way pulse bomb: a small ripple that seeds three targets and loops
start   SPL     #3
        MOV     #0, targeta
        MOV     #0, targetb
        MOV     #0, targetc
        JMP     start
targeta DAT     #0, #0
targetb DAT     #0, #0
targetc DAT     #0, #0
END
