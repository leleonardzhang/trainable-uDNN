/*
 * trainable.c
 */

#include "trainable.h"

int16_t mse_quad_cost(matrix *delta, uint16_t precision){
    /*
     * quad_cost = delta .^2
     */

    int16_t quad_cost = 0;
    uint16_t i;

    for (i = delta->numRows; i > 0; i--){
        quad_cost += fp_mul(delta->data[i - 1], delta->data[i - 1], precision);
    }

    return quad_cost;
}

matrix *mse_delta(matrix *delta, matrix *target, matrix *predict){
    /*
     * delta = target - predict
     */

    uint16_t i;
    for (i = target->numRows; i > 0; i --){
        delta->data[i - 1] = target->data[i - 1] - predict->data[i - 1];
    }
    return delta;
}

matrix *mse_kernel_gradient(matrix *gradient, matrix *delta, matrix *input, int16_t rate, uint16_t precision){
    /*
     * delta: n*1
     * input: m*1
     * gradient: n*m = delta * m^T
     */
    uint16_t gradient_numRows = gradient->numRows, gradient_numCols = gradient->numCols;
    uint16_t input_numRows = input->numRows, input_numCols = input->numCols;
    int16_t *gradient_data = gradient->data;

    input->numCols = input_numRows;
    input->numRows = input_numCols;

    gradient = matrix_multiply(gradient, delta, input, precision);

    input->numRows = input_numRows;
    input->numCols = input_numCols;

    uint16_t i;

    for (i = gradient_numRows * gradient_numCols; i > 0; i --){
        gradient_data[i - 1] = gradient_data[i - 1] >> rate;
    }

    return gradient;
}

matrix *mse_bias_gradient(matrix *bias_gradient, matrix *delta, int16_t rate, uint16_t precision){

    uint16_t bias_gradient_numRows = bias_gradient->numRows, bias_gradient_numCols = bias_gradient->numCols;

    uint16_t i;

    for (i = bias_gradient_numRows * bias_gradient_numCols; i > 0; i --){
        bias_gradient->data[i - 1] = delta->data[i - 1] >> rate;
    }

    return bias_gradient;
}



matrix *mse_gradient_descent(matrix *kernel, matrix *gradient, matrix *bias, matrix *bias_gradient, uint16_t m){

    uint16_t i;
    uint16_t kernel_length = kernel->numRows * kernel->numCols;
    uint16_t bias_length = bias->numRows * bias->numCols;

    for (i = kernel_length; i > 0; i --){
        kernel->data[i - 1] += (gradient->data[i - 1]  >> m);
    }

    for (i = bias_length; i > 0; i --){
        bias->data[i - 1] += (bias_gradient->data[i - 1] >> m);
    }

    return kernel;
}


matrix *mse_back_propagation(matrix *prev_delta, matrix *kernel, matrix *next_delta, uint16_t precision){
    /*
     * Back propagation to calculate previous delta
     * next_delta: n*1
     * prev_delta: m*1
     * kernel: n*m
     * Transpose next delta to 1*n and perform multiplication (1*n) * (n*m) = (1*m) and then transpose the result to m*1
     * prev_delta = (next_delta^T * kernel)^T
     */

    uint16_t next_delta_numRows = next_delta->numRows, next_delta_numCols = next_delta->numCols;
    uint16_t prev_delta_numRows = prev_delta->numRows, prev_delta_numCols = prev_delta->numCols;

    next_delta->numCols = next_delta_numRows;
    next_delta->numRows = next_delta_numCols;

    prev_delta->numCols = prev_delta_numRows;
    prev_delta->numRows = prev_delta_numCols;

    prev_delta = matrix_multiply(prev_delta, next_delta, kernel, precision);

    next_delta->numCols = next_delta_numCols;
    next_delta->numRows = next_delta_numRows;

    prev_delta->numCols = prev_delta_numCols;
    prev_delta->numRows = prev_delta_numRows;

    return prev_delta;
}


int16_t cross_entropy_loss(matrix *predict, uint16_t target, uint16_t precision){
    return -fp_ln(predict->data[target], TAYLOR_SERIES_ITERATIONS, precision);
}








