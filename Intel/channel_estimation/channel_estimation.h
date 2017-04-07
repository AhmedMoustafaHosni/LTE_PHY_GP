#pragma once


//  Function:    channe_estimation
//	Description : Generates channel estimation
//	Inputs : symb_0 - received DMRS number 1
//			 symb_1 - received DMRS number 2
//			 dmrs_0 - generated DMRS number 1
//			 dmrs_1 - generated DMRS number 2
//			 M_pusch_sc - number of subcarriers allocated to ue
//	Outputs : channel - channel estimation matrix to be used for equalization
//	By : Ahmed Moustafa



#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <fstream>
#include <iostream>
using namespace std;


void channe_estimation(MKL_Complex8* symb_0, MKL_Complex8* symb_1, MKL_Complex8* dmrs_0, MKL_Complex8* dmrs_1, const int M_pusch_sc, MKL_Complex8* channel);


#define N_sc_rb  12
#define N_symbs_per_slot 7
#define N_data_symbs_per_subframe 12	