#pragma once

#include <stdio.h>
#include <iostream>
#include <mkl.h>
#include <math.h>

using namespace std;

// number of subcarriers per resource block
#define N_SC_RB 12
#define M_PI_6  0.5235987756   // pi/6
#define M_PI       3.14159265358979323846   // pi
#define M_PI_4     0.785398163397448309616  // pi/4
#define N_ZC_RS_MAX 1193
#define M_RS_SC_MAX 1200


#define N_id_cell  2       // assume enodeB scheduled cell 2 for the UE
#define Nc  1600        // specified by the 36211 v8 standard for random sequence generation
#define n_s  0         // assume UE send on time slot 0
#define n_RNTI  10      // radio network temporary identifier given to the UE by enodeB(assume 10)

unsigned short* pseudo_random_sequence_gen(int, int);
void generate_dmrs(unsigned char N_suframe, unsigned int cell_id, unsigned char delta_ss, unsigned char cyclic_shift, unsigned char cyclic_shift_dci, unsigned char N_RB, MKL_Complex8*& dmrs_1, MKL_Complex8*& dmrs_2);