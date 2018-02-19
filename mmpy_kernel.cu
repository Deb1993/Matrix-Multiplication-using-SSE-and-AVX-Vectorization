// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "types.h"
using namespace std;
#define TW 32
#include <stdio.h>
__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
	//if(N%TW){

	//	int I =  blockIdx.y*blockDim.y + threadIdx.y;
	//	int J =  blockIdx.x*blockDim.x + threadIdx.x;

	//	if((I < N) && (J < N)){
	//		_DOUBLE_ _c = 0;
	//		for (unsigned int k = 0; k < N; k++) {
	//			_DOUBLE_ a = A[I * N + k];
	//			_DOUBLE_ b = B[k * N + J];
	//			_c += a * b;
	//		}
	//		C[I * N + J] = _c;
	//	}
	//}
	//else
	//{
	__shared__ double As[TW][TW], Bs[TW][TW];
	int ty = threadIdx.y, tx = threadIdx.x;
	int by = blockIdx.y, bx = blockIdx.x;
	int I = min(N-1,by*TW + ty); 
	double Cij = 0;
	int J= min(N-1,bx*TW + tx);
	for (int kk=0; kk<ceilf(float (N)/TW); kk++)
	{
		As[ty][tx] = A[I*N + kk*TW+tx];
		Bs[ty][tx] = B[(kk*TW+ty)*N + J];
		__syncthreads();
		//printf("Loaded A[index=%d]<-%f,B[index=%d]<-%f\t:\tbx=%d,by=%d,tx=%d,ty=%d,As=%f,Bs=%f\n",I*N + kk*TW+tx,A[I*N + kk*TW+tx],(kk*TW+ty)*N + J,B[(kk*TW+ty)*N + J],bx,by,tx,ty,As[ty][tx],Bs[ty][tx]);
	//	if(I<N && J<N)
	//	{
			for (int k=0; k<TW && k+kk*TW<N; k++)
			{
				Cij+= As[ty][k] * Bs[k][tx];
		//		printf("N=%d, I=%d,J=%d, bx=%d,by=%d, tx=%d,ty=%d,kk=%d,k=%d,As=%f,index=%d,Aaddr=%p,Bs=%f,index=%d,Baddr=%p,Cij=%f\n",N,I,J,bx,by,tx,ty,kk,k,As[ty][k],I*N+kk*TW+k,(void *)&As[ty][k],Bs[k][tx],(kk*TW+k)*N+J,(void *)&Bs[k][tx],Cij);
			}
	//	}
		__syncthreads();
	}
//	if(I<N && J<N)
//	{
		C[I*N + J] = Cij;
	//	printf("Storing Cij=%f to C[%d] as %f at %p\n",Cij,I*N+J,C[I*N+J],(void *)&C[I*N+J]);
//	}
	//}
}
