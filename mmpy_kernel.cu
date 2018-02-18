// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "types.h"
using namespace std;

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {

//    int I =  blockIdx.y*blockDim.y + threadIdx.y;
//    int J =  blockIdx.x*blockDim.x + threadIdx.x;
//
//    if((I < N) && (J < N)){
//        _DOUBLE_ _c = 0;
//        for (unsigned int k = 0; k < N; k++) {
//            _DOUBLE_ a = A[I * N + k];
//            _DOUBLE_ b = B[k * N + J];
//            _c += a * b;
//        }
//        C[I * N + J] = _c;
//    }
#define TW 32 

__shared__ double AS[TW][TW], BS[TW][TW];

int ty = threadIdx.y;
int tx = threadIdx.x;
int bx = blockIdx.x;
int by = blockIdx.y;

int I = by*TW + ty;
int J = bx*TW + tx;
double Cij = 0;

if( (I < N) && (J < N)) {
	for (int kk = 0 ; kk < N/TW ; kk++) {
		//printf("Inside kk loop = %d\n",kk);
		//printf("Inside kk loop = %d\n",kk);
		AS[ty][tx] = A[I*N + kk*TW + tx];
		BS[ty][tx] = B[(kk*TW + ty)*N  + J];
		__syncthreads();
		for(int k = 0; k < TW ; k++) {
			Cij += AS[ty][k] * BS[k][tx];
		//printf("Inside k loop = %d\n",k);
		 }
		__syncthreads();
		}
	C[I*N + J] = Cij;
	}
}
