#include <stdio.h>
#include <stdlib.h>

#include "timerc.h"

#define SIZE 1024*1024*16
#define gerror(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

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

__global__ void brute_force_refine(char *text, char *pattern, int *index, int *match, int number_of_blocks, int blocksize, int pattern_size){
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    
    if (id < number_of_blocks){
        int starting_pos = index[id*blocksize];
        int flag = 1;
        for (int i=0; i<pattern_size; i++){
            if (text[starting_pos+i] != pattern[i]){
                flag = 0;
                break;
            }
        }
        match[starting_pos] = flag;
    }
    
}

__global__ void brute_force_refine_blockoutput(char *text, char *pattern, int *blockoutput, int *match, int number_of_blocks, int blocksize, int pattern_size){
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    
    if (id < number_of_blocks){
        int flag = 1;
        int starting_pos = blockoutput[id];
        for (int i=0; i<pattern_size; i++){
            if (text[starting_pos+i] != pattern[i]){
                flag = 0;
            }
        }
        match[starting_pos] = flag;
    }
    
}


__global__ void nonperiodic_version_binary_tree_shared_memory(char *text, char *pattern, int *blockoutput, int *witness_array, int blocksize){
    int pid = threadIdx.x;
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    
    //int size = blocksize;
    int s = (blocksize + 2 - 1) / 2;
    int size = blocksize;
    
    __shared__ int buffer[1024];
    buffer[pid] = id;
    
    __syncthreads();
    
    while(s >= 2){
        if (threadIdx.x <= size/2){
            int starting_pos = pid;
            if (starting_pos + s >= size){
                buffer[pid] = id;
            }else{
                int j = buffer[starting_pos+s];
                int i = buffer[starting_pos];
                int k = witness_array[j-i];
                if (text[j+k] != pattern[k]){
                    buffer[starting_pos] = i;
                }else{
                    buffer[starting_pos] = j;
                }
            }
        }
        __syncthreads();
        s = (s+2-1)/2;
        size = (size + 2 - 1) / 2;
    }
     __syncthreads();
    if (threadIdx.x ==0){
        int j = buffer[1];
        int i = buffer[0];
        int k = witness_array[j-i];
        if (text[j + k ] != pattern[k]){
            blockoutput[blockIdx.x] = i;
        }else{
            blockoutput[blockIdx.x] = j;
        }
    }
}

__global__ void nonperiodic_version_binary_tree(char *text, char *pattern, int *output, int *witness_array, int blocksize){
    //get index
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    //use for looping
    int size = blocksize;
    int s = (blocksize + 2 - 1) / 2;
    //starting pos for the thread
    //create a dynamically large shared memory
    
    //read index to buffer
    output[id] = id;
    __syncthreads();
    
    while(s>=2){
        int starting_pos = id;
        if (threadIdx.x<size/2){
            if (starting_pos + s <size){
                int j = output[starting_pos+s];
                int i = output[starting_pos];
                int k = witness_array[j-i];
                if (text[output[j + k ]] == pattern[k]){
                    output[starting_pos] = j;
                }
            }
        }
        __syncthreads();
        s = (s+2-1)/2;
        size = (size + 2 - 1) / 2;
    }
    
    if (threadIdx.x ==0){
        int starting_pos = id;
        int j = output[starting_pos+s];
        int i = output[starting_pos];
        int k = witness_array[j-i];
        if (text[output[j + k ]] == pattern[k]){
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

void failure_function_cpu(char *pattern, int *failure_function, int pattern_size){
    
    failure_function[0] = 0;
    
    int k = 1;
    int j = 0;
    
    while ( k < pattern_size){
        if (pattern[k] == pattern[j]){
            j ++;
            failure_function[k] = j;
            k ++;
        }else{
            if (j !=0){
                k = failure_function[k-1];
            }else{
                failure_function[k] =0;
                k++;
            }
        }
    }
}

void serial_string_matching_KMP(char *text, char *pattern, int pattern_size, int text_size, int *failure_function){
    
    //index for text
    int i = 0;
    //index for pattern
    int j = 0;
    
    while (i < text_size){
        
        //if matching increment both index
        if (pattern[j] == text[i]){
            j++;
            i++;
        }
        
        //if match the pattern print message
        if (j == pattern_size){
            //printf("found at index %d \n", i-j);
            j = failure_function[j-1];
        }
        else if ( i < text_size && pattern[j] != text[i]){
            if (j != 0){
                j = failure_function[j-1];
            }else{
                i+=1;
            }
        }
        
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
    
    //intialized time
    float cpuTime;
    float gpuTime0;
    float gpuTime1;
    float gpuTime2;
    float gpuTime3;
    float cpuTime1;
    
    //read text to buffer
    while ((ch = getc(fp)) != EOF){
        text[size] = ch; 
        //match[size] = 0;
        size ++;
        if (size>=SIZE) break;
    }
    
    while ((ch =getc(fp2))!=EOF){
        pattern[pattern_size] = ch;
        pattern_size++;
    }
    
    size --;
    pattern_size--;
    printf("size %d \n", size);
    printf("pattern size %d \n", pattern_size);
    
    int *output = (int *) malloc (sizeof(int)*size);
    
    
    /*initialized match array*/
    match = (int *) malloc (size*sizeof(int));
    for (int i = 0; i < size; i++){
        match[i] = -1;
    }
    
    
    /*malloc wintess array*/
    int *witness_array = (int *)malloc(sizeof(int)*cap_division(pattern_size, 2));
    witness_array_cpu(pattern, witness_array, pattern_size);
    
    cstart();
    int *failure_function = (int *)malloc(sizeof(int)*(pattern_size));
    failure_function_cpu(pattern, failure_function, pattern_size);
    
    cend(&cpuTime);
    
    
    cstart();
    serial_string_matching_KMP(text, pattern, pattern_size, size, failure_function);
    cend(&cpuTime1);
    
    printf("CPU prepare time: %f", cpuTime);
    printf("KMP time: %f", cpuTime1);
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
    //block output
    int *dev_blockoutput;
    
     //config block and thread size
     //int number_of_threads = 32
    int number_of_threads = cap_division(pattern_size, 2);
    int number_of_blocks = (size + number_of_threads - 1) / number_of_threads;
    int *blockoutput = (int *) malloc(number_of_blocks*sizeof(int));
    
    
    gstart();
    cudaMalloc((void **)&dev_text, size*sizeof(char));
    cudaMalloc((void **)&dev_pattern, pattern_size*sizeof(char));
    cudaMalloc((void **)&dev_match, size*sizeof(int));
    //cudaMalloc((void **)&dev_output, sizeof(int)*size);
    cudaMalloc((void **)&dev_blockoutput, sizeof(int)*number_of_blocks);
    cudaMalloc((void **)&dev_witness, sizeof(int)*cap_division(pattern_size, 2));

    cudaMemcpy(dev_text, text, size*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_pattern, pattern, pattern_size*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_witness, witness_array, cap_division(pattern_size, 2)*sizeof(int), cudaMemcpyHostToDevice);
    
    gend(&gpuTime0);
    
    gstart();
    
    //brute_force<<<(size + 1024 - 1) / 1024, 1024>>>(dev_text, dev_pattern, dev_match, pattern_size, size);
    //nonperiodic_version_binary_tree<<<number_of_blocks, number_of_threads>>> (dev_text, dev_pattern, dev_output, dev_witness, number_of_threads);
    nonperiodic_version_binary_tree_shared_memory<<<number_of_blocks, number_of_threads>>> (dev_text, dev_pattern, dev_blockoutput, dev_witness, number_of_threads);
    
    
    cudaDeviceSynchronize();
    gend(&gpuTime1);
    
    gstart();
    
    //brute_force_refine<<<(size + 1024 - 1) / 1024, 1024>>>(dev_text, dev_pattern, dev_output, dev_match, number_of_blocks ,cap_division(pattern_size, 2), pattern_size);
    brute_force_refine_blockoutput<<<(size + 1024 - 1) / 1024, 1024>>>(dev_text, dev_pattern, dev_blockoutput, dev_match, number_of_blocks ,cap_division(pattern_size, 2), pattern_size);
    gend(&gpuTime2);
    
    gstart();
    cudaMemcpy(match, dev_match, size*sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(blockoutput, dev_blockoutput, size*sizeof(int), cudaMemcpyDeviceToHost);
    
    gend(&gpuTime3);
    
 
    /*
    printf("<<<<result>>>> \n");
    int flag =1;
    for (int i = 0; i< size; i++){
        printf("%d", match[i]);
    }
     /*
    if (flag ==1){
        printf("success");
    }else{
        printf ("error");
    }
    
    printf("\n");
    
    printf("<<<<output>>>> \n");
    for (int i = 0; i< number_of_blocks; i++){
        printf("%d ", blockoutput[i]);
    }
    printf("\n");
    */
    gerror( cudaPeekAtLastError() );
    cudaDeviceSynchronize();
    /*free memory*/
    cudaFree(dev_text);
    cudaFree(dev_pattern);
    cudaFree(dev_match);
    cudaFree(dev_output);
    cudaFree(dev_witness);
    cudaFree(dev_blockoutput);
    
    free(text);
    free(pattern);
    free(match);
    free(witness_array);
    free(failure_function);
    
    printf("CPUTIME: %f, GPUTIME0: %f, GPUTIME1: %f, GPUTIME2:%f, GPUTIME3:%f, TOTAL: %f", cpuTime,gpuTime0, gpuTime1, gpuTime2, gpuTime3, cpuTime+gpuTime1+gpuTime2 + gpuTime0+gpuTime3);
    
    //printf("CPUTIME: %f, GPUTIME0: %f, GPUTIME1: %f, GPUTIME3:%f, TOTAL: %f", cpuTime,gpuTime0, gpuTime1, gpuTime3, cpuTime+gpuTime1+gpuTime0+gpuTime3);
    
    

    
}
