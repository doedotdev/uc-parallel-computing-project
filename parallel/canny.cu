#include "utils.h"
#include  "math.h"
#define BLOCK_W 24
#define TILE_W 48
// inputChannel is GPU allocated for the R, G, or B channel 
// outputChannel is the same s input channel but is our result
// filter is our weight array, filterWidth specifies the width of the weight array



float  *d_filter;
unsigned char* d_dx;
unsigned char* d_dy;
unsigned char* d_outMag;
unsigned char* d_nmsGrid;
unsigned int* d_edgeMap;
void allocateMemoryAndCopyToGPU(const size_t numRows, const size_t numCols,
  const float* const h_filter, const size_t filterWidth)
{

  checkCudaErrors(cudaMalloc(&d_dx, sizeof(unsigned char)*numRows*numCols));
  checkCudaErrors(cudaMalloc(&d_dy, sizeof(unsigned char)*numRows*numCols));
  checkCudaErrors(cudaMalloc(&d_outMag, sizeof(unsigned char)*numRows*numCols));
  checkCudaErrors(cudaMemset(d_dx, 0, sizeof(unsigned char)*numRows*numCols));
  checkCudaErrors(cudaMemset(d_dy, 0, sizeof(unsigned char)*numRows*numCols));
  checkCudaErrors(cudaMemset(d_outMag, 0, sizeof(unsigned char)*numRows*numCols));
  checkCudaErrors(cudaMalloc(&d_filter, sizeof( float) * filterWidth * filterWidth));
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&d_nmsGrid, sizeof(unsigned char)*numRows*numCols));
  checkCudaErrors(cudaMemset(d_nmsGrid, 0, sizeof(unsigned char)*numRows*numCols));
  checkCudaErrors(cudaMalloc(&d_edgeMap, sizeof(unsigned int)*numRows*numCols));
  checkCudaErrors(cudaMemset(d_edgeMap, 0, sizeof(unsigned int)*numRows*numCols));
  
}

__global__
void gaussian_blur(const uchar4* const inputChannel,
                   uchar4* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
    // Red = x, Green = y, Blue = z
    __shared__ uchar4 img[TILE_W * TILE_W];
    const int R = filterWidth / 2;
    int centerX = blockIdx.x * blockDim.x + threadIdx.x - R;
    int centerY = blockIdx.y * blockDim.y + threadIdx.y - R;
    centerX = min(max(0, centerX), numCols - 1);
    centerY = min(max(0, centerY), numRows - 1);
    const int index = centerY * numCols + centerX;
    const int blockindex = threadIdx.y * blockDim.y + threadIdx.x;
    img[blockindex] = inputChannel[index];  
    __syncthreads();

  // Check if we have an out of bounds center
  if ( centerX >= numCols || centerY >= numRows )
   return;

  float outputPixRed = 0.0f;
  float outputPixGreen = 0.0f;
  float outputPixBlue = 0.0f;
  // Looping through filter rows and columns
      for (int filterRow = -R; filterRow < R; filterRow++)
      {
         for (int filterColumn = -R; filterColumn < R; filterColumn++)
         {
	    // Following from the example on how to clamp
	    // It looks like we are just making sure our location is not negative and not past our max columns or rows
	    int offsetX = threadIdx.x + filterColumn;
	    int offsetY = threadIdx.y + filterRow;
	    offsetX = min(max(0, offsetX), blockDim.x - 1);
	    offsetY = min(max(0, offsetY), blockDim.y- 1);
	    int offsetIdx = offsetY * blockDim.y + offsetX;
	    float baseValRed = static_cast<float>(img[offsetIdx].x);
	    float baseValGreen = static_cast<float>(img[offsetIdx].y);
	    float baseValBlue = static_cast<float>(img[offsetIdx].z);
	    float filterVal = static_cast<float>(filter[(filterRow + R) * filterWidth + (filterColumn + R)]);
	    outputPixRed += filterVal * baseValRed;
	    outputPixGreen += filterVal * baseValGreen;
	    outputPixBlue += filterVal * baseValBlue;
         }
      }
  outputChannel[index].x = outputPixRed;
  outputChannel[index].y = outputPixGreen;
  outputChannel[index].z = outputPixBlue;
  // NOTE: If a thread's absolute position 2D position is within the image, but some of
  // its neighbors are outside the image, then you will need to be extra careful. Instead
  // of trying to read such a neighbor value from GPU memory (which won't work because
  // the value is out of bounds), you should explicitly clamp the neighbor values you read
  // to be within the bounds of the image. If this is not clear to you, then please refer
  // to sequential reference solution for the exact clamping semantics you should follow.
}


__global__ void getIntensityGradient(const uchar4* const inputChannel, unsigned char* d_dx, unsigned char* d_dy, unsigned char* d_outMag, unsigned int numRow, unsigned int numCol)
{
    // First Derivative = f(x+1) - f(x-1)
    // Take each channel
     

    int dxR = 0;
    int dxG = 0;
    int dxB = 0;
    int dyR = 0;
    int dyG = 0;
    int dyB = 0;
    int globalIdxX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdxY = blockIdx.y * blockDim.y + threadIdx.y;
    int globalIdx = globalIdxY * numCol + globalIdxX;
    // Ignore top and bottom row
    if ((globalIdx > numCol) && globalIdx < (numRow*numCol - numCol)) 
    {
        // First Column
        if ( (globalIdx % numCol) == 0)
        {
           dxR = inputChannel[globalIdx+1].x - inputChannel[globalIdx].x;
           dxG = inputChannel[globalIdx+1].y - inputChannel[globalIdx].y;
           dxB = inputChannel[globalIdx+1].z - inputChannel[globalIdx].z;
           dyR = inputChannel[globalIdx+numCol].x - inputChannel[globalIdx].x;
           dyG = inputChannel[globalIdx+numCol].y - inputChannel[globalIdx].y;
           dyB = inputChannel[globalIdx+numCol].z - inputChannel[globalIdx].z;
        }
        else if ( (globalIdx % numCol) == (numCol - 1) )
        {
           dxR = inputChannel[globalIdx].x - inputChannel[globalIdx-1].x;
           dxG = inputChannel[globalIdx].y - inputChannel[globalIdx-1].y;
           dxB = inputChannel[globalIdx].z - inputChannel[globalIdx-1].z;
           dyR = inputChannel[globalIdx].x - inputChannel[globalIdx-numCol].x;
           dyG = inputChannel[globalIdx].y - inputChannel[globalIdx-numCol].y;
           dyB = inputChannel[globalIdx].z - inputChannel[globalIdx-numCol].z;
        }
        else
        {
           dxR = inputChannel[globalIdx + 1].x - inputChannel[globalIdx-1].x;
           dxG = inputChannel[globalIdx + 1].y - inputChannel[globalIdx-1].y;
           dxB = inputChannel[globalIdx + 1].z - inputChannel[globalIdx-1].z;
           dyR = inputChannel[globalIdx + numCol].x - inputChannel[globalIdx-numCol].x;
           dyG = inputChannel[globalIdx + numCol].y - inputChannel[globalIdx-numCol].y;
           dyB = inputChannel[globalIdx + numCol].z - inputChannel[globalIdx-numCol].z;
        }
    }
    d_dx[globalIdx] = (unsigned char).2989 * dxR + .587 * dxG + .114* dxB;
    d_dy[globalIdx] = (unsigned char).2989 * dyR + .587 * dyG + .114* dyB;
    d_outMag[globalIdx] = (unsigned char) sqrt( (double) d_dx[globalIdx]* d_dx[globalIdx]  + d_dy[globalIdx] * d_dy[globalIdx] );
}

__global__ void NonMaxSupression(unsigned char* d_Mag, unsigned char* dx, unsigned char* dy, unsigned char* d_nms, unsigned int numRow, unsigned int numCol)
{
   // We are dividing into 8 regions
   // Supress Pixel if in the gradient direction, it is not the max
   int globalIdx = (blockIdx.y * blockDim.y + threadIdx.y) * numCol + blockIdx.x * blockDim.x + threadIdx.x;
   float alpha;
   float gradient = atan2f( (float)dy[globalIdx], (float)dx[globalIdx]);
   float magA = 0;
   float magB = 0;

   if (globalIdx < numCol || globalIdx >= (numRow-1)*numCol || globalIdx % numCol == 0 || globalIdx % numCol == numCol - 1)
   {
      d_nms[globalIdx] = 0;
   }
   else if (gradient >= 0 && gradient < 45)
   {
      alpha = -1 * dy[globalIdx] / dx[globalIdx];
      magA = (1-alpha) * d_Mag[globalIdx + 1] + alpha * d_Mag[globalIdx - numCol + 1];
      magB = (1-alpha) * d_Mag[globalIdx - 1] + alpha * d_Mag[globalIdx - numCol - 1];
   }

   else if (gradient >= 45 && gradient < 90)
   {
      alpha = -1 * dx[globalIdx] / dy[globalIdx];
      magA = (1-alpha) * d_Mag[globalIdx + numCol] + alpha * d_Mag[globalIdx + numCol - 1];
      magB = (1-alpha) * d_Mag[globalIdx - numCol] + alpha * d_Mag[globalIdx + numCol + 1];
   }
   else if (gradient >= 90 && gradient <= 135)
   {
      alpha = dx[globalIdx] / dy[globalIdx];
      magA = (1-alpha) * d_Mag[globalIdx - numCol] + alpha * d_Mag[globalIdx - numCol - 1];
      magA = (1-alpha) * d_Mag[globalIdx + numCol] + alpha * d_Mag[globalIdx + numCol + 1];
   }
   else if (gradient >= 135 && gradient < 180)
   {
      alpha = dx[globalIdx] / dy[globalIdx];
      magA = (1 - alpha) * d_Mag[globalIdx - numCol] + alpha * d_Mag[globalIdx - numCol - 1];
      magB = (1 - alpha) * d_Mag[globalIdx + numCol] + alpha * d_Mag[globalIdx + numCol + 1];
   }
   else if (gradient >= 180 && gradient < 225)
   {
      alpha  = -1 * dy[globalIdx] / dx[globalIdx];
      magA = (1 - alpha) * d_Mag[globalIdx - 1] + alpha * d_Mag[globalIdx + numCol - 1];
      magB = (1 - alpha) * d_Mag[globalIdx + 1] + alpha * d_Mag[globalIdx - numCol + 1];
   }
   else if (gradient >= 225 && gradient < 270)
   {
      alpha = -1 * dx[globalIdx] / dy[globalIdx];
      magA = (1-alpha) * d_Mag[globalIdx + numCol] + alpha * d_Mag[globalIdx + numCol - 1];
      magB = (1-alpha) * d_Mag[globalIdx - numCol] + alpha * d_Mag[globalIdx - numCol + 1];
   }
   else if (gradient >=270 && gradient < 315)
   {
      alpha = dx[globalIdx] / dy[globalIdx];
      magA = (1 - alpha) * d_Mag[globalIdx + numCol] + alpha * d_Mag[globalIdx + numCol + 1];  
      magB = (1 - alpha) * d_Mag[globalIdx - numCol] + alpha * d_Mag[globalIdx - numCol - 1];      
   }
   else
   {
      alpha = dy[globalIdx] / dx[globalIdx];
      magA = (1 - alpha) * d_Mag[globalIdx + 1] + alpha*d_Mag[globalIdx + numCol + 1];
      magB = (1 - alpha) * d_Mag[globalIdx - 1] + alpha*d_Mag[globalIdx + numCol - 1];
   }

   if (d_Mag[globalIdx] < magA || d_Mag[globalIdx] < magB)
   {
      d_nms[globalIdx] = 0;
   }
   else
   {
      d_nms[globalIdx] = d_Mag[globalIdx];
   }
}

__global__ void hysteresis_highpass(unsigned char* d_mag, unsigned char* d_nms, unsigned int* d_map, unsigned int numRows, unsigned int numCols, unsigned char t_high)
{
   int globalIdx = (blockIdx.y * blockDim.y + threadIdx.y) * numCols + blockIdx.x * blockDim.x + threadIdx.x;
   if (globalIdx < numRows * numCols)
   {
      if (d_nms[globalIdx] >= t_high)
      {
	      d_map[globalIdx] = 1; //this is a strong edge
	      d_mag[globalIdx] = 255; // max for Uchar
      }
   }
}


__global__ void hysteresis_lowpass(unsigned char* d_mag, unsigned char* d_nms, unsigned int* d_map, unsigned int numRows, unsigned int numCols, unsigned char t_low)
{
   int globalIdx = (blockIdx.y * blockDim.y + threadIdx.y) * numCols + blockIdx.x * blockDim.x + threadIdx.x;
   if (globalIdx > numCols && globalIdx < numCols * numRows - numRows && (globalIdx % numCols > 0) && (globalIdx % numCols < numCols - 1))
   {
      if(d_map[globalIdx] == 1)
      {
         if (d_nms[globalIdx + 1] >= t_low)
            d_mag[globalIdx + 1] = 255;
	 else
            d_mag[globalIdx + 1] = 0;


         if(d_nms[globalIdx - 1] >= t_low)
            d_mag[globalIdx - 1] = 255;
	 else
	    d_mag[globalIdx - 1] = 0;

         if(d_nms[globalIdx + numCols] >= t_low)
            d_mag[globalIdx + numCols] = 255;
	 else
	    d_mag[globalIdx + numCols] = 0;	

         if (d_nms[globalIdx - numCols] >= t_low)
            d_mag[globalIdx - numCols] = 255;
	 else
            d_mag[globalIdx + numCols] = 0;

         if(d_nms[globalIdx - numCols - 1] >= t_low)
            d_mag[globalIdx - numCols - 1] = 255;
         else
            d_mag[globalIdx - numCols - 1] = 255;

         if(d_nms[globalIdx - numCols + 1] >= t_low)
            d_mag[globalIdx - numCols + 1] = 255;
         else
            d_mag[globalIdx - numCols + 1] = 0;

         if(d_nms[globalIdx + numCols - 1] >= t_low)
            d_mag[globalIdx + numCols - 1] = 255;
         else
            d_mag[globalIdx + numCols - 1] = 255;

         if(d_nms[globalIdx + numCols + 1] >= t_low)
            d_mag[globalIdx + numCols + 1] = 255;
         else
            d_mag[globalIdx + numCols + 1] = 0;  
      }
   }
}
/*
Entry Point for our serial program
*/
void cu_edge_detection(const uchar4* const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        const int filterWidth, unsigned char* h_finalImage)
{
  const dim3 blockSize(24, 24);
  const dim3 gridSize(numCols/blockSize.x + 1, numRows/blockSize.y + 1);
  unsigned char t_high = 100;
  unsigned char t_low = 10;
  gaussian_blur<<<gridSize, blockSize>>>(
      d_inputImageRGBA,
      d_outputImageRGBA,
      numRows,
      numCols,
      d_filter,
      filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  getIntensityGradient<<<gridSize, blockSize>>>(d_outputImageRGBA, d_dx, d_dy, d_outMag, numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  NonMaxSupression<<<gridSize, blockSize>>>(d_outMag, d_dx, d_dy, d_nmsGrid, numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  hysteresis_highpass<<<gridSize, blockSize>>>(d_outMag, d_nmsGrid, d_edgeMap, numRows, numCols, t_high);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  hysteresis_lowpass<<<gridSize, blockSize>>>( d_outMag, d_nmsGrid, d_edgeMap, numRows, numCols, t_low);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  cudaMemcpy(h_finalImage, d_outMag, numRows*numCols*sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

void cleanup()
{
  checkCudaErrors(cudaFree(d_dx));
  checkCudaErrors(cudaFree(d_dy));
  checkCudaErrors(cudaFree(d_outMag));
  checkCudaErrors(cudaFree(d_nmsGrid));
  checkCudaErrors(cudaFree(d_edgeMap));
}

