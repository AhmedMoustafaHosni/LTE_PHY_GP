#pragma once

//extern cudaStream_t stream_default;


#include <stdio.h>
#include <stdlib.h>

typedef unsigned char BYTE;

#define BUFF 100

BYTE* readBits(int argc, char* argv, int *numBits);