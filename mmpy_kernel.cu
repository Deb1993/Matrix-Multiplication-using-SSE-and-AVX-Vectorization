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
	double Cij_04 = 0;
	double Cij_08 = 0;
	double Cij_12 = 0;
	double Cij_16 = 0;
	double Cij_20 = 0;
	double Cij_24 = 0;
	double Cij_28 = 0;
	if(N)//%TW)
	{
		int I = by*TW + ty; 
		//int I = min(N-1,by*TW + ty); 
		//int J= min(N-1,bx*TW + tx);
		int J= bx*TW + tx;
		//	for (int kk=0; kk<ceilf(float (N)/TW); kk++)
		if((I < N) && (J < N)){
			#pragma unroll
		for (int kk=0; kk<(N/TW + int(bool(N%TW))); kk++)
			//for (int kk=0; kk<(N/TW + 1); kk++)
			{
			//printf("bx=%d,by=%d,tx=%d,ty=%d,Reading starting for kk=%d\n",bx,by,tx,ty,kk);
		//		As[ty][tx]    = __ldg(&A[I*N + kk*TW+tx]);
		//		As[ty+4][tx]  = __ldg(&A[(I+4)*N + kk*TW+tx]);
		//		As[ty+8][tx]  = __ldg(&A[(I+8)*N + kk*TW+tx]);
		//		As[ty+12][tx] = __ldg(&A[(I+12)*N + kk*TW+tx]);
		//		As[ty+16][tx] = __ldg(&A[(I+16)*N + kk*TW+tx]);
		//		As[ty+20][tx] = __ldg(&A[(I+20)*N + kk*TW+tx]);
		//		As[ty+24][tx] = __ldg(&A[(I+24)*N + kk*TW+tx]);
		//		As[ty+28][tx] = __ldg(&A[(I+28)*N + kk*TW+tx]);
		//		Bs[ty][tx]    = __ldg(&B[(kk*TW+ty)*N + J]);
		//		Bs[ty+4][tx]  = __ldg(&B[(kk*TW+ty+4)*N + J]);
		//		Bs[ty+8][tx]  = __ldg(&B[(kk*TW+ty+8)*N + J]);
		//		Bs[ty+12][tx] = __ldg(&B[(kk*TW+ty+12)*N + J]);
		//		Bs[ty+16][tx] = __ldg(&B[(kk*TW+ty+16)*N + J]);
		//		Bs[ty+20][tx] = __ldg(&B[(kk*TW+ty+20)*N + J]);
		//		Bs[ty+24][tx] = __ldg(&B[(kk*TW+ty+24)*N + J]);
		//		Bs[ty+28][tx] = __ldg(&B[(kk*TW+ty+28)*N + J]);
				As[ty][tx] = __ldg(&A[I*N + kk*TW+tx]);
				As[ty+4][tx] = (I+4 < N)? __ldg(&A[(I+4)*N + kk*TW+tx]):0;
				As[ty+8][tx] = (I+8 < N)? __ldg(&A[(I+8)*N + kk*TW+tx]):0;
				As[ty+12][tx] = (I+12 < N)? __ldg(&A[(I+12)*N + kk*TW+tx]):0;
				As[ty+16][tx] = (I+16 < N)? __ldg(&A[(I+16)*N + kk*TW+tx]):0;
				As[ty+20][tx] = (I+20 < N)? __ldg(&A[(I+20)*N + kk*TW+tx]):0;
				As[ty+24][tx] = (I+24 < N)? __ldg(&A[(I+24)*N + kk*TW+tx]):0;
				As[ty+28][tx] = (I+28 < N)? __ldg(&A[(I+28)*N + kk*TW+tx]):0;
				Bs[ty][tx] = __ldg(&B[(kk*TW+ty)*N + J]);
				Bs[ty+4][tx] = (kk*TW+ty+4 < N)? __ldg(&B[(kk*TW+ty+4)*N + J]):0;
				Bs[ty+8][tx] = (kk*TW+ty+8 < N)? __ldg(&B[(kk*TW+ty+8)*N + J]):0;
				Bs[ty+12][tx] = (kk*TW+ty+12 < N)? __ldg(&B[(kk*TW+ty+12)*N + J]):0;
				Bs[ty+16][tx] = (kk*TW+ty+16 < N)? __ldg(&B[(kk*TW+ty+16)*N + J]):0;
				Bs[ty+20][tx] = (kk*TW+ty+20 < N)? __ldg(&B[(kk*TW+ty+20)*N + J]):0;
				Bs[ty+24][tx] = (kk*TW+ty+24 < N)? __ldg(&B[(kk*TW+ty+24)*N + J]):0;
				Bs[ty+28][tx] = (kk*TW+ty+28 < N)? __ldg(&B[(kk*TW+ty+28)*N + J]):0;
				__syncthreads();
			//printf("bx=%d,by=%d,tx=%d,ty=%d,Reading completed for kk=%d\n",bx,by,tx,ty,kk);
				//for (int k=0; k<TW && k+kk*TW<N; k++)
			#pragma unroll
				for (int k=0; k<min(TW,N-kk*TW); k++)
				{
					Cij    += As[ty][k] * Bs[k][tx];
					Cij_04 += As[ty+4][k] * Bs[k][tx];
					Cij_08 += As[ty+8][k] * Bs[k][tx];
					Cij_12 += As[ty+12][k] * Bs[k][tx];
					Cij_16 += As[ty+16][k] * Bs[k][tx];
					Cij_20 += As[ty+20][k] * Bs[k][tx];
					Cij_24 += As[ty+24][k] * Bs[k][tx];
					Cij_28 += As[ty+28][k] * Bs[k][tx];
				}
				__syncthreads();
			//printf("bx=%d,by=%d,tx=%d,ty=%d,Computing completed for kk=%d\n",bx,by,tx,ty,kk);
			}
			//printf("bx=%d,by=%d,tx=%d,ty=%d,Store starting \n",bx,by,tx,ty);
			C[I*N + J]      = Cij;
			C[(I+4)*N + J]  = Cij_04;
			C[(I+8)*N + J]  = Cij_08;
			C[(I+12)*N + J] = Cij_12;
			C[(I+16)*N + J] = Cij_16;
			C[(I+20)*N + J] = Cij_20;
			C[(I+24)*N + J] = Cij_24;
			C[(I+28)*N + J] = Cij_28;
		//	if(I+4<N) C[(I+4)*N + J]  = Cij_04;
		//	if(I+8<N) C[(I+8)*N + J]  = Cij_08;
		//	if(I+12<N) C[(I+12)*N + J] = Cij_12;
		//	if(I+16<N) C[(I+16)*N + J] = Cij_16;
		//	if(I+20<N) C[(I+20)*N + J] = Cij_20;
		//	if(I+24<N) C[(I+24)*N + J] = Cij_24;
		//	if(I+28<N) C[(I+28)*N + J] = Cij_28;
			//printf("bx=%d,by=%d,tx=%d,ty=%d,Store completed\n",bx,by,tx,ty);
		}
	}
	else
	{
		int I = by*TW + ty; 
		int J = bx*TW + tx;
		//	for (int kk=0; kk<ceilf(float (N)/TW); kk++)

		if((I < N) && (J < N)){
			#pragma unroll
			for (int kk=0; kk<N/TW; kk++)
			{
				As[ty][tx]    = __ldg(&A[I*N + kk*TW+tx]);
				As[ty+4][tx]  = __ldg(&A[(I+4)*N + kk*TW+tx]);
				As[ty+8][tx]  = __ldg(&A[(I+8)*N + kk*TW+tx]);
				As[ty+12][tx] = __ldg(&A[(I+12)*N + kk*TW+tx]);
				As[ty+16][tx] = __ldg(&A[(I+16)*N + kk*TW+tx]);
				As[ty+20][tx] = __ldg(&A[(I+20)*N + kk*TW+tx]);
				As[ty+24][tx] = __ldg(&A[(I+24)*N + kk*TW+tx]);
				As[ty+28][tx] = __ldg(&A[(I+28)*N + kk*TW+tx]);
				Bs[ty][tx]    = __ldg(&B[(kk*TW+ty)*N + J]);
				Bs[ty+4][tx]  = __ldg(&B[(kk*TW+ty+4)*N + J]);
				Bs[ty+8][tx]  = __ldg(&B[(kk*TW+ty+8)*N + J]);
				Bs[ty+12][tx] = __ldg(&B[(kk*TW+ty+12)*N + J]);
				Bs[ty+16][tx] = __ldg(&B[(kk*TW+ty+16)*N + J]);
				Bs[ty+20][tx] = __ldg(&B[(kk*TW+ty+20)*N + J]);
				Bs[ty+24][tx] = __ldg(&B[(kk*TW+ty+24)*N + J]);
				Bs[ty+28][tx] = __ldg(&B[(kk*TW+ty+28)*N + J]);
				__syncthreads();
				//for (int k=0; k<TW && k+kk*TW<N; k++)
			#pragma unroll
				for (int k=0; k<TW; k++)
				{
					Cij    += As[ty][k] * Bs[k][tx];
					Cij_04 += As[ty+4][k] * Bs[k][tx];
					Cij_08 += As[ty+8][k] * Bs[k][tx];
					Cij_12 += As[ty+12][k] * Bs[k][tx];
					Cij_16 += As[ty+16][k] * Bs[k][tx];
					Cij_20 += As[ty+20][k] * Bs[k][tx];
					Cij_24 += As[ty+24][k] * Bs[k][tx];
					Cij_28 += As[ty+28][k] * Bs[k][tx];
				}
				__syncthreads();
			}
			C[I*N + J]      = Cij;
			C[(I+4)*N + J]  = Cij_04;
			C[(I+8)*N + J]  = Cij_08;
			C[(I+12)*N + J] = Cij_12;
			C[(I+16)*N + J] = Cij_16;
			C[(I+20)*N + J] = Cij_20;
			C[(I+24)*N + J] = Cij_24;
			C[(I+28)*N + J] = Cij_28;
		}
	}
}
