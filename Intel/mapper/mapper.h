#pragma once
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <mkl.h>


#define sqrt2 1.41421356237
#define sqrt2_rec 0.70710678118 
//#define NORM_64_6 0.9258200998   // 6/sqrt(42)
#define NORM_64_4 0.6172133998   // 4/sqrt(42)
#define NORM_64_3 0.4629100499   // 3/sqrt(42)
#define NORM_64_2 0.3086066999   // 2/sqrt(42)
//#define NORM_64_1 0.15430335     // 1/sqrt(42)

#define FRAME_LENGTH 86400
#define MOD 2                  // 2=QPSK, 4=16QAM, 6=64QAM

void mapper(float* bits, int bits_length, MKL_Complex8* symbols, char mod);

