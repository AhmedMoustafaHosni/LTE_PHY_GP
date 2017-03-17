#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>

#define N_id_cell  2       // assume enodeB scheduled cell 2 for the UE
#define Nc  1600        // specified by the 36211 v8 standard for random sequence generation
#define n_s  0         // assume UE send on time slot 4
#define n_RNTI  10      // radio network temporary identifier given to the UE by enodeB(assume 10)

unsigned short* pseudo_random_sequence_gen(int, int);