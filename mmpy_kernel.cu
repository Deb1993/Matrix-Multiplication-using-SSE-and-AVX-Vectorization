// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;
#define TW 32
//#define TWx  32
//#define TWy  8
#include <stdio.h>
__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
	__shared__ double As[TW][TW], Bs[TW][TW];
	int ty = threadIdx.y, tx = threadIdx.x;
	int by = blockIdx.y, bx = blockIdx.x;
	double Cij = 0;
	double Cij_4 = 0;
	double Cij_8 = 0;
	double Cij_12 = 0;
	double Cij_16 = 0;
	double Cij_20 = 0;
	double Cij_24 = 0;
	double Cij_28 = 0;
	if(N%TW)
{
	//ty = min(7,ty);
	int I = min(N-1,by*TW + ty); 
	int J= min(N-1,bx*TW + tx);

	
//	for (int kk=0; kk<ceilf(float (N)/TW); kk++)
	#pragma unroll
	for (int kk=0; kk<(N/TW + int(bool(N%TW))); kk++)
	{
		As[ty][tx] = A[I*N + kk*TW + tx];
		Bs[ty][tx] = B[(kk*TW+ty)*N + J];
		As[ty+4][tx] = A[(I+4)*N + kk*TW + tx];
		Bs[ty+4][tx] = B[(kk*TW+ty+4)*N + J];
		//printf("As = %f A = %f\n",As[ty][tx],A[I*N+kk*TWx+tx]); 
		//printf("ty = %d\n",ty);
		As[ty+8][tx] = A[(I+8)*N + kk*TW+tx];
		Bs[ty+8][tx] = B[(kk*TW+ty+8)*N + J];
		As[ty+12][tx] = A[(I+12)*N + kk*TW+tx];
		Bs[ty+12][tx] = B[(kk*TW+ty+12)*N + J];
		//printf("ty_8\n");
		As[ty+16][tx] = A[(I+16)*N + kk*TW+tx];
		Bs[ty+16][tx] = B[(kk*TW+ty+16)*N + J];
		As[ty+20][tx] = A[(I+20)*N + kk*TW+tx];
		Bs[ty+20][tx] = B[(kk*TW+ty+20)*N + J];
		//printf("ty_16\n");
		As[ty+24][tx] = A[(I+24)*N + kk*TW+tx];
		Bs[ty+24][tx] = B[(kk*TW+ty+24)*N + J];
		As[ty+28][tx] = A[(I+28)*N + kk*TW+tx];
		Bs[ty+28][tx] = B[(kk*TW+ty+28)*N + J];
		//printf("ty_24\n");
		__syncthreads();
			//for (int k=0; k<TW && k+kk*TW<N; k++)
			#pragma unroll
			for (int k=0; k<min(TW,N-kk*TW); k++)
			{
				Cij+= As[ty][k] * Bs[k][tx];
				Cij_4+= As[ty+4][k] * Bs[k][tx];
				Cij_8+= As[ty+8][k] * Bs[k][tx];
				Cij_12+= As[ty+12][k] * Bs[k][tx];
				Cij_16+= As[ty+16][k] * Bs[k][tx];
				Cij_20+= As[ty+20][k] * Bs[k][tx];
				Cij_24+= As[ty+24][k] * Bs[k][tx];
				Cij_28+= As[ty+28][k] * Bs[k][tx];
			}
		__syncthreads();
	}
		C[I*N + J] = Cij;
		C[(I+4)*N + J] = Cij_4;
		C[(I+8)*N + J] = Cij_8;
		C[(I+12)*N + J] = Cij_12;
		C[(I+16)*N + J] = Cij_16;
		C[(I+20)*N + J] = Cij_20;
		C[(I+24)*N + J] = Cij_24;
		C[(I+28)*N + J] = Cij_28;
}
else
{
	int I = by*TW + ty; 
	int J = bx*TW + tx;
//	for (int kk=0; kk<ceilf(float (N)/TW); kk++)
	if((I < N) && (J < N)) { 
	#pragma unroll
	for (int kk=0; kk<N/TW; kk++)
	{
		As[ty][tx] = A[I*N + kk*TW + tx];
		Bs[ty][tx] = B[(kk*TW+ty)*N + J];
		As[ty+4][tx] = A[(I+4)*N + kk*TW + tx];
		Bs[ty+4][tx] = B[(kk*TW+ty+4)*N + J];
		//printf("As = %f A = %f\n",As[ty][tx],A[I*N+kk*TWx+tx]); 
		//printf("ty = %d\n",ty);
		As[ty+8][tx] = A[(I+8)*N + kk*TW+tx];
		Bs[ty+8][tx] = B[(kk*TW+ty+8)*N + J];
		As[ty+12][tx] = A[(I+12)*N + kk*TW+tx];
		Bs[ty+12][tx] = B[(kk*TW+ty+12)*N + J];
		//printf("ty_8\n");
		As[ty+16][tx] = A[(I+16)*N + kk*TW+tx];
		Bs[ty+16][tx] = B[(kk*TW+ty+16)*N + J];
		As[ty+20][tx] = A[(I+20)*N + kk*TW+tx];
		Bs[ty+20][tx] = B[(kk*TW+ty+20)*N + J];
		//printf("ty_16\n");
		As[ty+24][tx] = A[(I+24)*N + kk*TW+tx];
		Bs[ty+24][tx] = B[(kk*TW+ty+24)*N + J];
		As[ty+28][tx] = A[(I+28)*N + kk*TW+tx];
		Bs[ty+28][tx] = B[(kk*TW+ty+28)*N + J];
		__syncthreads();
			//for (int k=0; k<TW && k+kk*TW<N; k++)
			#pragma unroll
			for (int k=0; k<TW; k++)
			{
				Cij+= As[ty][k] * Bs[k][tx];
				Cij_4+= As[ty+4][k] * Bs[k][tx];
				Cij_8+= As[ty+8][k] * Bs[k][tx];
				Cij_12+= As[ty+12][k] * Bs[k][tx];
				Cij_16+= As[ty+16][k] * Bs[k][tx];
				Cij_20+= As[ty+20][k] * Bs[k][tx];
				Cij_24+= As[ty+24][k] * Bs[k][tx];
				Cij_28+= As[ty+28][k] * Bs[k][tx];
			}
		__syncthreads();
		}
		C[I*N + J] = Cij;
		C[(I+4)*N + J] = Cij_4;
		C[(I+8)*N + J] = Cij_8;
		C[(I+12)*N + J] = Cij_12;
		C[(I+16)*N + J] = Cij_16;
		C[(I+20)*N + J] = Cij_20;
		C[(I+24)*N + J] = Cij_24;
		C[(I+28)*N + J] = Cij_28;
		}
	}
}
