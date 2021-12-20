/*
 * main.c
 * Include processing inputs, outputs and applying deep learning models
 */

#include "main.h"
static uint16_t i = 0, k = 0, j = 0, m = 0;

//#pragma PERSISTENT(LAYER_0)
//static layer_t LAYER_0 = {.class=Conv2d, .activation=&fp_relu, .kernel=&MNIST_KERNEL_0_MAT, .bias=&MNIST_BIAS_0_MAT,
//                                                     .numChannels=1, .numFilters=8, .padding=Same, .stride_numCols=1, .stride_numRows=1};
//
//#pragma PERSISTENT(LAYER_1)
//static layer_t LAYER_1 = {.class=Maxpooling, .activation=&fp_linear, .numChannels=8, .padding=Valid, .stride_numCols=3, .stride_numRows=3, .pool_numCols=3, .pool_numRows=3};
//
//#pragma PERSISTENT(LAYER_2)
//static layer_t LAYER_2 = {.class=Conv2d, .activation=&fp_relu, .kernel=&MNIST_KERNEL_1_MAT, .bias=&MNIST_BIAS_1_MAT,
//                                                     .numChannels=8, .numFilters=16, .padding=Same, .stride_numCols=1, .stride_numRows=1};
//
//#pragma PERSISTENT(LAYER_3)
//static layer_t LAYER_3 = {.class=Maxpooling, .activation=&fp_linear, .numChannels=16, .padding=Valid, .stride_numCols=3, .stride_numRows=3, .pool_numCols=3, .pool_numRows=3};
//
//#pragma PERSISTENT(LAYER_4)
//static layer_t LAYER_4 = {.class=Flatten, .numChannels=16};
//
//#pragma PERSISTENT(LAYER_5)
//static layer_t LAYER_5 = {.class=Dense, .activation=&fp_relu, .kernel=&MNIST_KERNEL_2_MAT, .bias=&MNIST_BIAS_2_MAT};
//
//#pragma PERSISTENT(LAYER_6)
//static layer_t LAYER_6 = {.class=Dense, .activation=&fp_relu, .kernel=&MNIST_KERNEL_3_MAT, .bias=&MNIST_BIAS_3_MAT};
//
//#pragma PERSISTENT(LAYERS)
//static layer_t *LAYERS[7] = {&LAYER_0, &LAYER_1, &LAYER_2, &LAYER_3, &LAYER_4, &LAYER_5, &LAYER_6};
//
//#pragma PERSISTENT(MODEL)
//static model_t MODEL = {.numLayers=7, .layers=LAYERS, .input=&MNIST_INPUT_MAT, .output=&MNIST_OUTPUT_MAT};


void main(void){

    /* stop watchdog timer */
    WDTCTL = WDTPW | WDTHOLD;

    /* initialize GPIO System */
    init_gpio();

    /* initialize the clock and baudrate */
    init_clock_system();

//    evaluate(128,256);
//    mnist_test();

    for (i = 0; i < 50; i ++){

        memset(KERNEL_GRADIENT, 0, sizeof(dtype) * 320);
        memset(BIAS_GRADIENT, 0, sizeof(dtype) * 10);
        LOSS = 0;

        for (j = 0; j < 128; j ++){

            memcpy(INPUT, &X[(j << 5)], 32);
            memcpy(&TARGET, &Y[j], 1);

            dense(&OUTPUT_MAT, &INPUT_MAT, &KERNEL_MAT, &BIAS_MAT, &fp_linear, FIXED_POINT_PRECISION);

            sparsemax(&ACTIVATION_MAT, &OUTPUT_MAT, FIXED_POINT_PRECISION);

            LOSS += (sparsemax_loss(&ACTIVATION_MAT, TARGET, 1, FIXED_POINT_PRECISION) >> 4);

            sparsemax_kernel_gradient(&KERNEL_GRADIENT_TEMP_MAT, &ACTIVATION_MAT, &INPUT_MAT, TARGET, 3, FIXED_POINT_PRECISION);
            sparsemax_bias_gradient(&BIAS_GRADIENT_TEMP_MAT, &ACTIVATION_MAT, TARGET, 3, FIXED_POINT_PRECISION);

            matrix_add(&KERNEL_GRADIENT_MAT, &KERNEL_GRADIENT_MAT, &KERNEL_GRADIENT_TEMP_MAT);
            matrix_add(&BIAS_GRADIENT_MAT, &BIAS_GRADIENT_MAT, &BIAS_GRADIENT_TEMP_MAT);

        }

        matrix_neg(&KERNEL_GRADIENT_MAT, &KERNEL_GRADIENT_MAT, FIXED_POINT_PRECISION);
        matrix_neg(&BIAS_GRADIENT_MAT, &BIAS_GRADIENT_MAT, FIXED_POINT_PRECISION);

        gradient_descent(&KERNEL_MAT, &KERNEL_GRADIENT_MAT, &BIAS_MAT, &BIAS_GRADIENT_MAT, 7);
        LOSS = LOSS >> 3;
        __no_operation();

    }

    for (j = 128; j < 160; j ++){

        memcpy(INPUT, &X[(j << 5)], 32);
        memcpy(&TARGET, &Y[j], 1);

        dense(&OUTPUT_MAT, &INPUT_MAT, &KERNEL_MAT, &BIAS_MAT, &fp_linear, FIXED_POINT_PRECISION);
        LABEL = argmax(&OUTPUT_MAT);
        __no_operation();
    }


}

//void train(){
//    for (j = 0; j < 50; j ++){
//        memset(LAYER_0_KERNEL_GRADIENT, 0, 1024 * sizeof(dtype));
//        memset(LAYER_1_KERNEL_GRADIENT, 0, 320 * sizeof(dtype));
//        memset(LAYER_0_BIAS_GRADIENT, 0, 32 * sizeof(dtype));
//        memset(LAYER_1_BIAS_GRADIENT, 0, 10 * sizeof(dtype));
//        AVG_LOSS = 0;
//
//        for (i = 0; i < 960; i +=32){
//            memset(TARGET, 0, 10 * sizeof(dtype));
//            TARGET[Y[i]] = 1024;
//            memset(LAYER_0_KERNEL_GRADIENT_TEMP, 0, 1024 * sizeof(dtype));
//            memset(LAYER_1_KERNEL_GRADIENT_TEMP, 0, 320 * sizeof(dtype));
//            memset(LAYER_0_BIAS_GRADIENT_TEMP, 0, 32 * sizeof(dtype));
//            memset(LAYER_1_BIAS_GRADIENT_TEMP, 0, 10 * sizeof(dtype));
//
//            for (k = 0; k < 32; k ++){
//                input_buffer[k] = X[(i << 5) + k];
//            }
//
//            dense(&LAYER_0_OUTPUT_MAT, &INPUT_MAT, &LAYER_0_KERNEL_MAT,  &LAYER_0_BIAS_MAT, &fp_linear, FIXED_POINT_PRECISION);
//
//            dense(&LAYER_1_OUTPUT_MAT, &LAYER_0_OUTPUT_MAT, &LAYER_1_KERNEL_MAT,  &LAYER_1_BIAS_MAT, &fp_linear, FIXED_POINT_PRECISION);
//
//            mse_delta(&LAYER_1_LOSS_MAT, &TARGET_MAT, &LAYER_1_OUTPUT_MAT);
//
//            LOSS = mse_quad_cost(&LAYER_1_LOSS_MAT, FIXED_POINT_PRECISION);
//
//            AVG_LOSS += LOSS >> 1;
//
//            mse_back_propagation(&LAYER_0_LOSS_MAT, &LAYER_1_KERNEL_MAT, &LAYER_1_LOSS_MAT, FIXED_POINT_PRECISION);
//
//            mse_kernel_gradient(&LAYER_1_KERNEL_GRADIENT_TEMP_MAT, &LAYER_1_LOSS_MAT, &LAYER_0_OUTPUT_MAT, 1, FIXED_POINT_PRECISION);
//            matrix_add(&LAYER_1_KERNEL_GRADIENT_MAT, &LAYER_1_KERNEL_GRADIENT_MAT, &LAYER_1_KERNEL_GRADIENT_TEMP_MAT);
//
//            mse_bias_gradient(&LAYER_1_BIAS_GRADIENT_TEMP_MAT, &LAYER_1_LOSS_MAT, 1, FIXED_POINT_PRECISION);
//            matrix_add(&LAYER_1_BIAS_GRADIENT_MAT, &LAYER_1_BIAS_GRADIENT_MAT, &LAYER_1_BIAS_GRADIENT_TEMP_MAT);
//
//            mse_kernel_gradient(&LAYER_0_KERNEL_GRADIENT_TEMP_MAT, &LAYER_0_LOSS_MAT, &INPUT_MAT, 0, FIXED_POINT_PRECISION);
//            matrix_add(&LAYER_0_KERNEL_GRADIENT_MAT, &LAYER_0_KERNEL_GRADIENT_MAT, &LAYER_0_KERNEL_GRADIENT_TEMP_MAT);
//
//            mse_bias_gradient(&LAYER_0_BIAS_GRADIENT_TEMP_MAT, &LAYER_0_LOSS_MAT, 0, FIXED_POINT_PRECISION);
//            matrix_add(&LAYER_0_BIAS_GRADIENT_MAT, &LAYER_0_BIAS_GRADIENT_MAT, &LAYER_0_BIAS_GRADIENT_TEMP_MAT);
//
//            __no_operation();
//       }
//
//       AVG_LOSS = AVG_LOSS >> 4;
//       gradient_descent(&LAYER_1_KERNEL_MAT, &LAYER_1_KERNEL_GRADIENT_MAT, &LAYER_1_BIAS_MAT, &LAYER_1_BIAS_GRADIENT_MAT,  5);
//       gradient_descent(&LAYER_0_KERNEL_MAT, &LAYER_0_KERNEL_GRADIENT_MAT, &LAYER_0_BIAS_MAT, &LAYER_0_BIAS_GRADIENT_MAT,  5);
//       __no_operation();
//    }
//}
//
//void mnist_test(){
//    predict(&MODEL);
//}








