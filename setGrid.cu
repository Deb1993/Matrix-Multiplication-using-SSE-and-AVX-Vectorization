void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{
   // set your block dimensions and grid dimensions here
   gridDim.x = n / blockDim.x;
   gridDim.y = n / blockDim.y;
   //gridDim.x = 24;
   //gridDim.y = 16;
   if(n % blockDim.x != 0)
   	gridDim.x++;
   if(n % blockDim.y != 0)
    	gridDim.y++;
}
