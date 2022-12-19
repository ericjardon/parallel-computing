#include <math.h>


void sigmoid(float x) {
    return (1. / (1 + exp(-x)));
};

/* FORWARD PASS */
void convLayer_forward(int M, int C, int H, int W, int K, float* X, float* W, float *Y) {
    // Assumes filters are square, KxK
    int m, c, h, w, p ,q;

    int H_out = H-K+1; // output dimensions
    int W_out = W-K+1; // output dimensions

    for (m=0; m<M; m++) {   // For every output feature map m
        
        // For every output entry (h,w)
        for (h=0; h<H_out; h++) {
            for (w=0; w<W_out; w++) {
                           
                Y[m, h, w] = 0; // initial value of sum

                // Compute weighted sum over C input feature maps (channels)
                for (c=0; c<C; c++) {
                    //For k*k entries using input c and filter (m,c)
                    for (p=0; p<K; p++) {
                        for (q=0; q<K; q++) {
                            Y[m, h, w] += X[c, h+p, w+q] * W[m, c, p, q];
                        }
                    }
                }
            }
        }
    }
}


void poolingLayer_forward(int M, int H, int W, int K, float* Y, float* S, float* b) {
    // Y is our input array
    // S is our output array
    
    int m,h,w,x,y,p,q;

    for(m=0; m<M; m++) {       // for every output feature map
        unsigned int out_height = H/K;
        for (h=0; h < out_height; h++) { // h is x coordinate
        unsigned int out_width = W/K;
            for (w=0; w < W; w++) {     // w is y coordinate

                // Initialize average
                S[m, h, w] = 0;

                for (p=0; p < K; p++) {
                    for (q=0; q < K; q++) {
                        // sum KxK entries, average by multiplying by 1/N
                        S[m, h, w] += Y[m, K*h + p, K*w + q] / (K*K);
                    }
                }

                // add bias and apply non linear activation
                S[m, h, w] = sigmoid(S[m,h,w] + b[m]);
                // note: every feature map m has an associated bias
            }
        }
    }
}

/* BACKPROPAGATION */
/*
 The fully connexted layer is simply Y = W*X (no convolution)
 The gradient of the loss function is given by two things: dE/dX and dE/dW
 Per the chain rule ( since E(Y(W,X)) )
 dE/dX = W.t * dE/dY
  and
 dE/dW = dE/dY * X.t

 which makes sense!

 The reason we compute dE/dX is to backpropagate
 but the gradient we use for updating our weights is dE/dW

*/


/**
 * @brief Computes gradient of Loss with respect to the inputs X of this layer.
 * X is a volume, or an array of bidimensional feature maps.
 * @param M Number of output feature maps to this layer
 * @param C Number of input feature maps to this layer
 * @param H_in Height of input feature maps to this layer
 * @param W_in Width of input feature maps to this layer
 * @param K Size of filter kernel
 * @param dE_dY Pointer to the array of computed partial derivatives from next layer
 * @param W Array of weights of the current layer
 * @param dE_dX Pointer to the allocated array for the gradient of the Loss function with respect to this layer's inputs
 */
void convLayer_backward_xgrad(int M, int C, int H_in, int W_in, int K, float* dE_dY, float* W, float* dE_dX) {
    // Every layer has C inputs, M outputs (feature maps).
    // The gradient of Loss function with respect to an input feature map X (just one channel)
    // is the sum of backward convolutions of W transpose over the layer's outputs
    int m,c,h,w,p,q;

    int H_out = H_in-K+1;
    int W_out = W_in-K+1;

    // Initialize every partial derivative
    for (c=0; c<C; c++) {
        for (h=0; h<H_in; h++) {
            for (w=0; w<W_in; w++) {
                dE_dX[c, h, w] = 0.;
            }
        }
    }

    // Use next layer's gradient to compute this layer's gradient

    for (m=0; m<M; m++) {    // for every entry (h,w) of every output map m
        for (h=0; h<H_in; h++) { 
            for (w=0; w<W_in; w++) {
                
                // Compute KxK partial derivatives
                for (c=0; c<C; c++) {
                    for (p=0; p < K; p++) {
                        for (q=0; q < K; q++) {
                            // Multiply gradient of next layer with this gradient's weights
                            dE_dX[c, h+p, w+q] += dE_dY[m, h, w] * W[m, c, p, q];
                        }
                    }
                }
            }
        }
    }   
}

void convLayer_backward_wgrad(int M, int C, int H, int W, int K, float* dE_dY, float* X, float* dE_dW) {
    int m, c, h, w, p, q;

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    // Init all partial derivatives to zero
    for (m=0; m<M; m++) {
        for (c=0; c<C; c++) {
            for (p=0; p<K; p++) {
                for (q=0; q<K; q++) {
                    dE_dW[m , c, p, q] = 0.;
                }
            }
        }
    }

    for (m=0; m<M; m++) {
        for (c=0; c<C; c++) {
            
            // Since every kernel W[c,m] affects all elements of the m-th output map Y[m],
            // we must sum the gradients over all the pixels in the output feature map
            for (h=0; h<H_out; h++) {
                for (w=0; w<W_out; w++) {

                    for (p=0; p<K; p++) {
                        for (q=0; q<K; q++) {
                            // chain rule: dE/dw_mcpq = Sigma[h,w](X_chw*(dE/dy_mchw))
                            dE_dW[m, c, p, q] += X[c, h+p, w+q] * dE_dY[m, c, h, w];
                        }
                    }
                
                }
            }
        }
    }

}


int main() {

    printf("CNN sequential implementation");

    return 0;
}