.text
    .align 4
    .globl main
main:
     pushl %ebp
     movl %esp,%ebp
     xorl %eax,%eax
     leave
     ret