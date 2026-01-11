
        ;minimal self-branching loop: spawns two copies each cycle
        ;creative, simple core-walker
        ORG     start
start   SPL     2, 0
        JMP     start
        JMP     start
        END     start
