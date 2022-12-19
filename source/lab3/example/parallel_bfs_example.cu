
#define QUEUE_SIZE 4096
#define BLOCK_SIZE 256
#define BLOCK_QUEUE_SIZE 1024

/*
PARALLELIZED BFS

A work-efficient implementation is toparallelize each iteration of the while-not-empty-queue-loop
of a BFS by having multiple threads to collaboratively process the previous queue (frontier) array 
and assemble the current queue array [LWH 2010]. 
This effectively parallelizes the outer for-loop that goes over all vertices in the queue in a given level.

Ping-pong buffering uses two same-sized queues, a "previous" and a "current".
The previous is used for reading (popping) purposes, while the current is for
constructing (appending, or inserting) purposes. 
On every iteration, these two arrays swap places.

p_tail is our read-only queue's last pointer
c_tail is our write-only queue's last pointer
every iteration, they switch and we restart the new c_tail (ping-pong buffering)
*/

__global__ void bfs_kernel(unsigned int* p_frontier, unsigned int* p_frontier_tail, unsigned int* c_frontier, unsigned int* c_frontier_tail,
    unsigned int* edges, unsigned int* dest, unsigned int* labels, unsigned int* visited) {

        __shared__ unsigned int c_frontier_s[BLOCK_QUEUE_SIZE];
        __shared__ unsigned int c_frontier_tail_s, our_c_frontier_tail;

        if (threadIdx.x == 0){
            c_frontier_tail_s = 0;
        }
        __syncthreads();    // do not proceed until all threads have reached this point.
        
        const unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x; // row*width + col

        if (tid < *p_frontier_tail) {   

            const unsigned int my_vertex = p_frontier[tid];

            for (unsigned int i = edges[my_vertex]; i < edges[my_vertex+1]; ++i) {
                const unsigned int was_visited = atomicExch(&(visited[dest[i]]), 1);
                if (!wasvisited) {
                    label[dest[i]] = label[my_vertex] + 1;

                    const unsigned int my_tail = atomicAdd(&c_frontier_tail, 1);
                    
                    if (my_tail < BLOCK_QUEUE_SIZE) {
                        c_frontier_s[my_tail] = dest[i];
                    } else {
                        c_frontier_tail_s = BLOCK_QUEUE_SIZE;
                        const unsigned int my_global_tail = atomicAdd(c_frontier_tail, 1);
                        c_frontier[my_global_tail] = dest[i];
                    }
                }
            }
        }
        __syncthreads();    // wait for all threads
        if (threadIdx.x == 0) {
            our_c_frontier_tail = atomicAdd(c_frontier_tail, c_frontier_tail_s);
        }

        __syncthreads();    // wait for all threads
        for (unsigned int i = threadIdx.x; i < c_frontier_tail_s; i += blockDim.x) {
            c_frontier[our_c_frontier_tail + i] = c_frontier_s[i];
        }
}


void bfs(unsigned int souce, unsigned int *edges, unsigned int *dest, unsigned int *label) {
    // allocate: edges_d, dest_d, label_d and visited_d in device global memory
    // copy edges, dest and label into corresponding global memory
    // allocate frontier_d, c_frontier_tail_d, p_frontier_tail_d

    unsigned int *c_frontier_d = &frontier_d[0];
    unsigned int *p_frontier_d = &frontier_d[QUEUE_SIZE];

    // Kernel to initialize in the device global memory
    // visited -- all elements to 0 except source
    // c tail to 0, p tail to 1
    // label[source] = 0;

    p_frontier_tail = 1; // host

    while (p_frontier_tail > 0) {
        int num_blocks = ceil(p_frontier_tail / float(BLOCK_SIZE));

        bfs_kernel<<<num_blocks, BLOCK_SIZE>>>(p_frontier_d, p_frontier_tail_d, c_frontier_d, c_frontier_tail_d, edges_d, dest_d, label_d, visited_d);
        // copy c_frontier_tail_d to host into p_frontier_tail for the while loop test
        // equivalent to reading the number of elements in queue to process in next iteration

        // ping-pong buffering
        int *temp = c_frontier_d; 
        c_frontier_d = p_frontier_d;
        p_frontier_d = temp;
        // kernel to set p_tail <- c_tail and c_tail=0;
    }
}