void print_asm(char *arg1,long int size){ 
    __asm__ (
				"mov $1, %%rax\n"
        "mov %0, %%rsi\n" 
        "mov $1, %%rdi\n"
        "mov %1, %%rdx\n" 
        "syscall"
    ::"m" (arg1),"m" (size) 
    );
}
int main(void) {
    char *d="ss\n";
    print_asm(d,3);
return 0;
}