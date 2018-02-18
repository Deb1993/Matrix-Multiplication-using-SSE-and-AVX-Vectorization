// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;
#define TW 32
__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
	if(N%TW){

	int I =  blockIdx.y*blockDim.y + threadIdx.y;
	int J =  blockIdx.x*blockDim.x + threadIdx.x;

	if((I < N) && (J < N)){
		_DOUBLE_ _c = 0;
		for (unsigned int k = 0; k < N; k++) {
			_DOUBLE_ a = A[I * N + k];
			_DOUBLE_ b = B[k * N + J];
			_c += a * b;
		}
		C[I * N + J] = _c;
	}
}
else
{
	__shared__ double As[TW][TW], Bs[TW][TW];
	int ty = threadIdx.y, tx = threadIdx.x;
	int by = blockIdx.y, bx = blockIdx.x;
	int I = by*TW + ty; 
	double Cij = 0;
	int J= bx*TW + tx;
	for (int kk=0; kk<N/TW; kk++){
		As[ty][tx] = A[I*N + kk*TW+tx];
		Bs[ty][tx] = B[(kk*TW+ty)*N + J];
		__syncthreads();
		for (int k=0; k<TW; k++)
			Cij+= As[ty][k] * Bs[k][tx];
		__syncthreads();
	}
	C[I*N + J] = Cij;
}
