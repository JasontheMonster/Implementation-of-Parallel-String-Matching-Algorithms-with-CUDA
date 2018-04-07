#include <stdio.h>
#include <stdlib.h>


#define SIZE 1024
/*use pattern to compare starting with every possible position*/
__global__ void brute_force(char *text, char *pattern, int *match, int pattern_size, int text_size){

        /*get the absolute pid*/
        int pid = threadIdx.x + blockIdx.x*blockDim.x;

        if (pid <= text_size - pattern_size){
        
            int flag = 1; 
            for (int i = 0; i < pattern_size; i++){
                if (text[pid+i] != pattern[i]){
                        flag = 0;
                }
            }
            match[pid] = flag;
        }
}

//__global__ void brute_force_refine(char *text, char *pattern, int *index, ){
  //  for (int i = 0; )
    
    
//}

__global__ void nonperiodic_version_binary_tree(char *text, char *pattern, int *output, int *witness_array, int blocksize){
    //get index
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    //use for looping
    int size = blocksize;
    int s = 1;
    //starting pos for the thread
    //create a dynamically large shared memory
    
    //read index to buffer
    output[id] = id;
    __syncthreads();
    
    while(size!=2){
        int starting_pos = id*2*s;
        if (threadIdx.x<size/2){
            if (starting_pos + s >=size){
                output[id] = starting_pos;
            }else{
                int k = witness_array[output[starting_pos+s]-output[starting_pos]];
                if (text[output[starting_pos+s] + k ] != pattern[k]){
                    output[starting_pos] = output[starting_pos];
                }else{
                    output[starting_pos] = output[starting_pos+s];
                }
        }
    }
        __syncthreads();
        s = s *2;
        size = (size + 2 - 1) / 2;
    }
    if (threadIdx.x ==0){
        int starting_pos = id*2*s;
        int k = witness_array[output[starting_pos+s]-output[starting_pos]];
        if (text[output[starting_pos+s] + k ] != pattern[k]){
            output[starting_pos] = output[starting_pos];
        }else{
            output[starting_pos] = output[starting_pos+s];
        }
        
    }
   // output[blockIdx.x] = buffer[blockIdx.x * blockDim.x];
}

int cap_division(int x, int y){
    return (x + y - 1) / y;
}
/*CPU version of wintess array calculation*/
void witness_array_cpu(char *pattern, int *witness_array, int pattern_size){
    if (pattern_size >2){
        witness_array[0] = 0;
        for (int i = 1; i<cap_division(pattern_size, 2); i++){
            for (int j=0; j<cap_division(pattern_size, 2); j++){
                if (pattern[j] != pattern[i+j]){
                    witness_array[i] = j;
                    break;
                }
            }
        }
    }else{
        witness_array[0] = 0;
    }
}

int main(){

     /*initialization; 
       open file
       read file char by char and store in heap
       */
     FILE *fp;
    FILE *fp2;
     char ch;
     fp = fopen("test.txt", "r");
    fp2 = fopen("pattern.txt", "r");
    
     char * text = (char *) malloc (SIZE*sizeof(char)); //size text buffer for text
    char * pattern = (char *) malloc (SIZE*sizeof(char));
    
     int * match; //size text buffeer for match array
     int size = 0;
     int pattern_size = 0;
     //int blocksize = 32;
    
     //read text to buffer
     while ((ch = getc(fp)) != EOF){
        text[size] = ch; 
        //match[size] = 0;
        size ++;
     }
    
    while ((ch =getc(fp2))!=EOF){
        pattern[pattern_size] = ch;
        pattern_size++;
    }
    size --;
    pattern_size--;
    int *output = (int *) malloc (sizeof(int)*size);
    
    match = (int *) malloc (size*sizeof(int));
    
    /*malloc wintess array*/
    int *witness_array = (int *)malloc(sizeof(int)*cap_division(pattern_size, 2));
    witness_array_cpu(pattern, witness_array, pattern_size);
    
    printf("pattern array: \n");
    for (int i = 0; i < cap_division(pattern_size, 2); i++){
        printf("%d ", witness_array[i]);
    }
    
    
    /* GPU init*/
     //text buffer in device
     char *dev_text;
     //pattern buffer in device
     char *dev_pattern;
     // match buffer in device
     int *dev_match;
    //output buffer in device
    int *dev_output;
    //witness array
    int *dev_witness;
    
     //config block and thread size
     //int number_of_threads = 32
    int number_of_threads = cap_division(pattern_size, 2);
    int number_of_blocks = (size + number_of_threads - 1) / number_of_threads;
    

     cudaMalloc((void **)&dev_text, size*sizeof(char));
     cudaMalloc((void **)&dev_pattern, pattern_size*sizeof(char));
     cudaMalloc((void **)&dev_match, size*sizeof(int));
    cudaMalloc((void **)&dev_output, sizeof(int)*size);
    cudaMalloc((void **)&dev_witness, sizeof(int)*cap_division(pattern_size, 2));

     cudaMemcpy(dev_text, text, size*sizeof(char), cudaMemcpyHostToDevice);
     cudaMemcpy(dev_pattern, pattern, pattern_size*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_witness, witness_array, cap_division(pattern_size, 2)*sizeof(int), cudaMemcpyHostToDevice);
     //cudaMemcpy(dev_match, match, size*sizeof(int), cudaMemcpyHostToDevice);
    
     
     //brute_force<<<1, size>>>(dev_text, dev_pattern, dev_match, pattern_size, size);
     nonperiodic_version_binary_tree<<<number_of_blocks, number_of_threads>>> (dev_text, dev_pattern, dev_output,
                                                                               dev_witness, number_of_threads);
    
     cudaMemcpy(output, dev_output, size*sizeof(int), cudaMemcpyDeviceToHost);
    
     printf("<<<<result>>>> \n");
     for (int i = 0; i< size; i+=cap_division(pattern_size, 2)){
         printf("%d ", output[i]);
     }
     printf("\n");
    
    
    

    
      




}
