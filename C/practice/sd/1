#define _POSIX_C_SOURCE 200809L
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
    setlinebuf();
    fprintf()
    return 0;
    
}
