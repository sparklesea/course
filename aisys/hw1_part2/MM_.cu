#include <cuda.h>
#define BLOCKNUM 32

__global__ void MatMul_Kernel(int *M,int *N,int *P,int width){
//计算线程id

//计算输出矩阵的每个元素值

}

void MatMul(int *M,int *N,int *P,int width){
// 分配内存


//内存拷贝

//执行kernel "MatMul_Kernel"

//内存拷贝

//释放内存
}
