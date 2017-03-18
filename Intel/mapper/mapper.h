#pragma once
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <cmath>
#include <mkl.h>


#define sqrt2 1.41421356237
#define sqrt2_rec 0.70710678118 

#define FRAME_LENGTH 86400

void mapper(float* bits, int bits_length, MKL_Complex8* symbols, char mod);

