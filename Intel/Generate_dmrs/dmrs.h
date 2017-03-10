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

void generate_dmrs(unsigned char N_suframe, unsigned int cell_id, unsigned char delta_ss, unsigned char cyclic_shift, unsigned char cyclic_shift_dci, unsigned char N_RB, MKL_Complex8*& dmrs_1, MKL_Complex8*& dmrs_2);