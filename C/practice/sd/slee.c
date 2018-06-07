#include <stdio.h>

#include <unistd.h>

void main(){
    for(int i=0;i<10;i++){
 sleep(1);
printf("dewdew%d\r",i);
fflush(stdout);
}
}