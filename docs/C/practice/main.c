#include <xmmintrin.h>
void print_asm(char *arg1,long int size){ 
    __asm__ (
        "vpabsw ymm1, ymm0"
        "movaps %0, %%xmm0\n"
		"mov $1, %%rax\n"
        "mov %0, %%rsi\n" 
        "mov $1, %%rdi\n"
        "mov %1, %%rdx\n" 
        "syscall"
    ::"m" (arg1),"m" (size) 
    );
}
int main(void) {
    char *d="ss";
    while(1){
        
    }
    print_asm(d,3);
return 0;
}