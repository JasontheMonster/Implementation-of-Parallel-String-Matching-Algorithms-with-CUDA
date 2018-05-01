#include<stdio.h>
#include<stdlib.h>
#include <ctype.h>

#define SIZE 1024
int main(){
  FILE *fp = fopen("test.txt", "w");
  if (fp ==NULL){
    printf("FILE open error");
    exit(1);
  }
    for (int i =0; i < SIZE; i++){
      if (i % 3 ==0){
         fputs("A", fp);
      }else{
        fputs("B", fp);
      }
  }

  fclose(fp);
}
