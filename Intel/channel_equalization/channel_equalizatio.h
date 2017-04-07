#pragma once

/*

Function:		 channel_equalization_zf
Description : equalise the channel effecct on the received signal


Inputs :	  modulated_subframe - received subframe without demodulation signal in one vector
channel - estimated channel


Outputs :    equalised symbols

// By : Ahmed Moustafa

*/


#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <fstream>
#include <iostream>
using namespace std;


void channel_equalization(MKL_Complex8* modulated_subframe, MKL_Complex8* channel, MKL_Complex8* equalized_subframe, const int M_pusch_sc);


#define N_sc_rb  12
#define N_symbs_per_slot 7
#define N_data_symbs_per_subframe 12	