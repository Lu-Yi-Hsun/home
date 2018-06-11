# Mix c & ASM
## 簡介

為了加快效能與特殊需求可以使用

組合語言與c語言混和撰寫方式

os:linux(x64)

編譯器:gcc

接下來皆以64位元linux示範

與32位元會有所不一樣
## Linux system call
[第二手資源](http://blog.rchapman.org/posts/Linux_System_Call_Table_for_x86_64/)

### 第一手資料尋找方式

[linux原始碼](https://github.com/torvalds/linux)

以下舉例：sys_write

* 從[syscall_64.tbl](https://github.com/torvalds/linux/blob/master/arch/x86/entry/syscalls/syscall_64.tbl#L12)尋找sys_write

    根據該檔案知道sys_write system call編號1 所以暫存器%rax 填入1

    

* 尋找實做該system call的檔案
    sys_write實做檔案

    [linux system call 實做位置](http://www.jollen.org/blog/2006/10/linux_26_system_call12.html#c2)
    
    linux/fs/read_write.c

* 根據該檔案搜尋實做的函數 
    
    [實做函數位置](https://github.com/torvalds/linux/blob/master/fs/read_write.c#L607)
    
    ```
    linux/fs/read_write.c 
    ```
    x:編號不一定

    要搜尋有write的SYSCALL_DEFINEx

    ```
    SYSCALL_DEFINEx(write,....){...}
    ```
    

!!! __x64_sys_write
    [source](https://github.com/torvalds/linux/blob/master/fs/read_write.c#L607)</br>
    我們看到ksys_write(fd, buf, count)有三個變數</br>
    根據ABI規則依序填入暫存器fd=%rdi,buf=%rsi,count=%rdx
    ```c
    SYSCALL_DEFINE3(write, unsigned int, fd, const char __user *, buf,size_t, count)
    {
        return ksys_write(fd, buf, count);
    }
    ```
根據上面答案建立此表格

表格1-1

%rax|System call|%rdi|%rsi|%rdx|%r10|%r8|%r9|
:----|:----|:----|:----|:----|:----|:----|:----
1|__x64_sys_write|fd|buf|count

關於此段指令為什麼填入1的理由

["mov $1, %%rdi\n"](https://stackoverflow.com/questions/5256599/what-are-file-descriptors-explained-in-simple-terms/5257718#5257718)

```c
void print_asm(char *arg1,long int size){ 
    __asm__ (
        "mov $1, %%rax\n"//system call 編碼
        "mov $1, %%rdi\n"//fd 設定1 代表把字串輸入/dev/stdout 這裡就是螢幕輸出地方
        "mov %0,%%rsi\n"//輸入字串記憶體位置
        "mov %1, %%rdx\n" //這裡輸入字串長度 ,可以跟記憶體位置搭配來輸出到螢幕
        "syscall"//x64 要用此呼叫systemcall 不能在使用int $0x80
        :
        :"m" (arg1),"m" (size) //詳細請參考gcc inline asm
    );
}
```

 

## 輸出範例

### C內嵌ASM
實做直接呼叫linux system code

!!! print
    ```c
    void print_asm(char *arg1,long int size){ 
        __asm__ (
            "mov $1, %%rax\n"//system call 編碼
            "mov $1, %%rdi\n"//fd 設定1 代表把字串輸入/dev/stdout 這裡就是螢幕輸出地方
            "mov %0,%%rsi\n"//輸入字串記憶體位置
            "mov %1, %%rdx\n" //這裡輸入字串長度 ,可以跟記憶體位置搭配來輸出到螢幕
            "syscall"//x64 要用此呼叫systemcall 不能在使用int $0x80
            :
            :"m" (arg1),"m" (size) //詳細請參考gcc inline asm
        );
    }
    int main(void) {
        char *d="ss\n";
        print_asm(d,3);
    return 0;
    }
    ```
### C & ASM混和編譯