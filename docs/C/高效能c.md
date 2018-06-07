# 高效能c
 
## 前製作業

### 測試平台規格 
規格|參數
:----|:----
Compiler|gcc (GCC) 8.1.0
Architecture:|x86_64
CPU op-mode(s):|32-bit, 64-bit
Byte Order:|Little Endian
CPU(s):|4
On-line CPU(s) list:|0-3
Thread(s) per core:|2
Core(s) per socket:|2
Socket(s):|1
NUMA node(s):|1
Vendor ID:|GenuineIntel
CPU family:|6
Model:|61
Model name:|Intel(R) Core(TM) i7-5500U CPU @ 2.40GHz
Stepping:|4
CPU MHz:|2702.991
CPU max MHz:|3000.0000
CPU min MHz:|500.0000
BogoMIPS:|4790.41
Virtualization:|VT-x
L1d cache:|32K
L1i cache:|32K
L2 cache:|256K
L3 cache:|4096K
NUMA node0 CPU(s):|0-3
Flags:|fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cplvmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb invpcid_single pti tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid rdseed adx smapintel_pt xsaveopt ibpb ibrs stibp dtherm ida arat pln pts

### 測試時間程式

!!! timespec 
    tv_sec:1970到現在的秒數

    tv_nsec:在tv_sec該秒數內又走了多少奈秒

    1970到現在的奈秒=tv_sec*10e9+tv_nsec

```c
struct timespec 
{
    time_t   tv_sec;        /* seconds */
    long     tv_nsec;       /* nanoseconds */
};
```

??? code
    ```c
    #include <time.h>
    #include <stdio.h>

    double diff_in_sec(struct timespec,struct timespec);
    long diff_in_nsec(struct timespec,struct timespec);

    double diff_in_sec(struct timespec t1, struct timespec t2)
    {
        long diff=diff_in_nsec(t1,t2);
        return (diff/10e9);
    }

    long diff_in_nsec(struct timespec start, struct timespec end)
    {
        return (end.tv_sec*10e9+end.tv_nsec)-(start.tv_sec*10e9+start.tv_nsec); 
    }

    int main()
    {
        struct timespec start_time,end_time;

        clock_gettime(CLOCK_REALTIME, &start_time);
        for(int i=0;i<100000000;i++){
            //Do Something
        }
        clock_gettime(CLOCK_REALTIME, &end_time);

        printf("resolution: %.10lfs\n",diff_in_sec(start_time,end_time));
        printf("resolution: %ldns",diff_in_nsec(start_time,end_time));
        return 0;
    }
    ```
---
## Memory

### padding

gcc will auto padding in 32bit

padding for 64bit system
```c
struct foo {
int x;
char xx[1];
int y;  
}__attribute__((align(64)));
```

gcc without padding  
```c
struct foo {
int x;
char xx[1];
int y;  
}__attribute__((packed));
```

---

## Cache

### False Sharing

!!! False_sharing
    ```c
    struct foo {
        int x;
        //char xx[60];
        int y;  
        };
    ```
!!! sharing
    ```c
    struct foo {
        int x;
        char xx[60];
        int y;  
        };
    ```

??? full_code
    ```c
    #include <time.h>
    #include <stdio.h>
    #include <pthread.h>

    double diff_in_sec(struct timespec,struct timespec);
    long diff_in_nsec(struct timespec,struct timespec);

    double diff_in_sec(struct timespec t1, struct timespec t2)
    {
        long diff=diff_in_nsec(t1,t2);
        return (diff/10e9);
    }

    long diff_in_nsec(struct timespec start, struct timespec end)
    {
        return (end.tv_sec*10e9+end.tv_nsec)-(start.tv_sec*10e9+start.tv_nsec); 
    }

    struct foo {
    int x;
    //__________________
    //char xx[60];
    //__________________
    int y;  
    };//__attribute__((align(64)));__attribute__((packed));
    static struct foo f;
    int sum_a(void)
    {
        int s = 0;
        int i;
        for (i = 0; i < 1000000000; ++i)
            s += f.x;
        return s;
    }
    void inc_b(void)
    {
        int i;
        for (i = 0; i < 1000000000; ++i)
            ++f.y;
    }
    
    t(){
    pthread_t id;
    pthread_create(&id,NULL,(void *) sum_a,NULL);
    pthread_t id2;  
    pthread_create(&id2,NULL,(void *) inc_b,NULL);
    pthread_join(id,NULL);
    pthread_join(id2,NULL);
    }
    /*TODO:mode*/
    o(){
    sum_a();
    inc_b();
    }
    void main(){
    printf("%d\n",sizeof(f));
    struct timespec start_time,end_time;
    clock_gettime(CLOCK_REALTIME, &start_time);
    t();
    clock_gettime(CLOCK_REALTIME, &end_time);
    printf("%.10lfs\t%ldns",diff_in_sec(start_time,end_time),diff_in_nsec(start_time,end_time));


    }

    ```
