// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;
#define TW 32
#include <stdio.h>
__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
	__shared__ double As[TW][TW], Bs[TW][TW];
	int ty = threadIdx.y, tx = threadIdx.x;
	int by = blockIdx.y, bx = blockIdx.x;
	double Cij = 0;
	if(N%TW)
{
	int I = min(N-1,by*TW + ty); 
	int J= min(N-1,bx*TW + tx);
//	for (int kk=0; kk<ceilf(float (N)/TW); kk++)
	for (int kk=0; kk<(N/TW + int(bool(N%TW))); kk++)
	{
		As[ty][tx] = A[I*N + kk*TW+tx];
		Bs[ty][tx] = B[(kk*TW+ty)*N + J];
		__syncthreads();
			//for (int k=0; k<TW && k+kk*TW<N; k++)
			for (int k=0; k<min(TW,N-kk*TW); k++)
			{
				Cij+= As[ty][k] * Bs[k][tx];
			}
		__syncthreads();
	}
		C[I*N + J] = Cij;
}
else
{
	int I = by*TW + ty; 
	int J= bx*TW + tx;
//	for (int kk=0; kk<ceilf(float (N)/TW); kk++)
	for (int kk=0; kk<N/TW; kk++)
	{
		As[ty][tx] = A[I*N + kk*TW+tx];
		Bs[ty][tx] = B[(kk*TW+ty)*N + J];
		__syncthreads();
			//for (int k=0; k<TW && k+kk*TW<N; k++)
			for (int k=0; k<TW; k++)
			{
				Cij+= As[ty][k] * Bs[k][tx];
			}
		__syncthreads();
		}
		C[I*N + J] = Cij;

	}
}
