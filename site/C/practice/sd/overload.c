#include<stdio.h>


struct Struct1{

    int x;
    int dd;
    char ddd[22];
    int y;
};

struct Struct2{

    int x;
    int y;
};
void foo(void * arg1){

struct Struct2 *struct2PtrVar= arg1;
printf("%d",struct2PtrVar->y);
 
 
}

void main(void ){
struct Struct1 s;
s.x=3;
s.dd=322;
s.y=34444;
foo(&s);

}