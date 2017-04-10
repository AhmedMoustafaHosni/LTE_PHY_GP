/*
*  LTE siso chain
*  Max timing: 1.12 ms
*
*  Merge by Mohammed Osama
*/

#include "Intel_siso.h"


int main()
{
	unsigned char subframe_number = 0;		  // frame number
	unsigned char n_s = subframe_number * 2;  // slot number
	unsigned int cell_ID = 2;
	unsigned char delta_ss = 0;
	unsigned char cyclic_shift = 0;
	unsigned char cyclic_shift_dci = 0;
	
	
	// intialize Data and RI_bits
	float* bits = (float*)calloc(INTRLV, sizeof(float));  // input bits
	float* ri_bits = (float*)malloc(N_RI_bits * MOD * sizeof(float));

	for (int i = 0; i <= INTRLV - 1; i += 4)
	{
		bits[i] = 1;
		bits[i + 1] = 1;
		bits[i + 2] = 0;
		bits[i + 3] = 1;
	}
	for (int i = 0; i <= N_RI_bits * MOD - 1; i += 4)
	{
		ri_bits[i] = 0;
		ri_bits[i + 1] = 1;
		ri_bits[i + 2] = 0;
		ri_bits[i + 3] = 0;

	}

	/**********************    1st call      **********************/
	MKL_Complex8* pusch_bb = (MKL_Complex8*)malloc(30720 * sizeof(MKL_Complex8*));
	sisoTx(pusch_bb, bits, INTRLV, ri_bits, N_RI_bits * MOD, MOD, RESOURCE_BLOCKS, subframe_number, cell_ID, delta_ss, cyclic_shift, cyclic_shift_dci);
	/**************************************************************/
	free(pusch_bb);

	double s_initial = 0, s_elapsed = 0;
	s_initial = dsecnd();
	/**********************    2nd call      **********************/
	pusch_bb = (MKL_Complex8*)malloc(30720 * sizeof(MKL_Complex8*));
	sisoTx(pusch_bb, bits, INTRLV, ri_bits, N_RI_bits * MOD, MOD, RESOURCE_BLOCKS, subframe_number, cell_ID, delta_ss, cyclic_shift, cyclic_shift_dci);
	/**************************************************************/
	s_elapsed = (dsecnd() - s_initial);
	printf(" \n \n tx chain is completed at %.5f milliseconds \n\n", (s_elapsed * 1000));

	return 0;

}
