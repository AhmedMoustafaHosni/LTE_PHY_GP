#pragma once

#include <stdio.h>
#include <stdlib.h>

typedef unsigned char Byte;

#define BUFF 100

Byte* readBits(int argc, char* argv, int *numBits);
