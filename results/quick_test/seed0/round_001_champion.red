
;name copycat
;author ChatGPT
;description A tiny self-replicating "copycat" warrior. It splits into two processes and attempts to mirror its own behavior as it runs.
;strategy basic self-replication test

ORG     start
start   SPL     #2, >start
        MOV     #1, >start
        JMP     start
        DAT     #0, #0
END
