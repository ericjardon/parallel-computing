/*
Computation patterns of CNN are compute-intensive 
and highly parallel. 

== Parallel Strategy
We can compute in parallel:
* SGD mini batch samples (N in parallel, where N = data set size / batch size)
* Output feature maps (M in parallel)
* Output feature map pixels (H*W in parallel)

Note that this strategy uses 4 levels of parallelism.
Total num of parallelized iterations is N*M*H_out*W_out.

== Kernel Design:
We organize threads in a 3D grid (arrangement of thread blocks).
- 1d (X) corresponds to N mini-batches of data set
- 2d (Y) corresponds to M output feature maps
- 3d (Z) corresponds to the location of the tile (patch) position of output feature map

Z depends on the number of tiles we do in horizontal and vertical directions
of the output feature map.

Assuming W_out and H_out are multiples of tile width (e.g. 16).

Each block will be responsible for a single tile.
*/


#define TILE_WIDTH 16

// Incredibly high parallelism but high global memory consumption
// execution speed is limited by global memory bandwidth
// How can we reduce traffic to global memory? With memory TILING!
__global__ void ConvLayerForward_SimpleKernel(int C, int W_grid, int K, float* X, float* W, float* Y) {
    int n, m, h, w, c, p, q;

    n = blockIdx.x; // mini batch index
    m = blockIdx.y; // output feature map index

    h = blockIdx.z / W_grid + threadIdx.y;
    w = blockIdx.z % W_grid + threadIdx.x;

    float wSum = 0.;

    for (c=0; c<C; c++) { // for every input feature map (channel)
        // Convolution: k*k weighted elements at pivot h,w
        for (p=0; p<K; p++) {
            for (q=0; q<K; q++) {
                wSum = wSum + X[n, c, h+p, w+q] * W[m, c, p, q];
            }
        }
    }

    // Output pixel at h,w
    Y[n, m, h, w] = wSum;
}

__global__ void ConvLayerForward_Kernel(int C, int W_grid, int K, float* X, float* W, float* Y) {
    int n, m, h, w, c, p, q;

    n = blockIdx.x; // mini batch index
    m = blockIdx.y; // output feature map index

    h = blockIdx.z / W_grid + threadIdx.y;
    w = blockIdx.z % W_grid + threadIdx.x;

    float wSum = 0.;

    for (c=0; c<C; c++) { // for every input feature map (channel)
        // Convolution: k*k weighted elements at pivot h,w
        for (p=0; p<K; p++) {
            for (q=0; q<K; q++) {
                wSum = wSum + X[n, c, h+p, w+q] * W[m, c, p, q];
            }
        }
    }

    // Output pixel at h,w
    Y[n, m, h, w] = wSum;
}

/* 
MEMORY-TILED KERNEL 

1) Load filter W[m,c] to memory (global).

2) All threads should collaborate to copy the portion X[n,c,.,.] 
into the shared array X_shared, required for computing Output tile.
This should be a KxK portion of X

3) We compute the partial sum of output Y_shared[n, m, ., .]
* what is a partial sum?

4) Move on to next input channel C
*/

__global__ void ConvLayerForward_Kernel(int C, int W_grid, int K, float* X, float* W, float* Y) {
    // Uses memory tiling among blocks
    int n, m, h0, w0, h_base, w_base, h, w;
    int X_tile_width = TILE_WIDTH + K - 1;
    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];
    float* W_shared = &shmem[X_tile_width * X_tile_width];

    n = blockIdx.x; // mini batch index
    m = blockIdx.y; // output feature map index

    h0 = threadIdx.x; // row
    w0 = threadIdx.y; // col

    h_base = (blockIdx.z / W_grid) * TILE_WIDTH; // vertical base of output data used for the block
    w_base = (blockIdx.z % W_grid) * TILE_WIDTH; // horizontal base of output data used for the block

    h = h_base + h0;
    w = w_base + w0;

    float wSum = 0.;     // weighted sum over all input channels
    int c, i, j, p, q;

    for (c=0; c < C; c++) {


        if ((h0 < K) && (w0 < K))
            // Load weights in shmem
            W_shared[h0, w0] = W[m, c, h0, w0];
            // h0, w0 shorthand for thread x, y
        
        __syncthreads();
        // all threads have now loaded in their share of W
        int max_h = h_base + X_tile_width;
        int max_w = w_base + X_tile_width;
        
        for (i=h; i < max_h; i+= TILE_WIDTH) {
            for (j = w; j < max_w; j += TILE_WIDTH) {
                X_shared[i-h_base,j-w_base] = X[n, c, h, w];
            }
        }   // load tile from X[n, c, .,.] into memory, splitting work among TILE_WIDTH*TILE_WIDTH threads
        __syncthreads();
        // all threads have now loaded their share of X

        // Convolution of KxK elements
        for (p=0; p<K; p++) {
            for (q=0; q<K; q++) {
                wSum = wSum + X_shared[h+p, w+q] * W_shared[p, q];
            }
        }
        __syncthreads(); // all must have finished using shared memory
    }   

    Y[n , m, h, w] = wSum;
} // reference: fig 16.3

int main() {
    W_grid = W_out/TILE_WIDTH; // # horizontal tiles per output map 
    H_grid = H_out/TILE_WIDTH; // # vertical tiles per output map
    Z = H_grid * W_grid;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(N, M, Z);
    ConvLayerForward_Kernel<<<gridDim, blockDim>>>(...);
    // TODO: clean code
}


