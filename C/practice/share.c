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