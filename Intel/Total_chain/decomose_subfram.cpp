/*******************************************************************************
* Function:    decompose_subframe
* Description: gride demapping
* Inputs:      gride_in  - output from SCFDMA demodulator
*			   RB        - Number of RBs assigned to the ue
*
* Outputs:	   dmrs1     - Demodulated referance  signal 1
*              dmrs1     - Demodulated referance  signal 2
*			   data      - Data Symbols to transform predecoder
*
*
* by: Khaled Ahmed Ali
********************************************************************************/

#include "Intel_siso.h"

void decompose_subframe(MKL_Complex8 ** gride_in, MKL_Complex8 * data, MKL_Complex8* dmrs_1, MKL_Complex8* dmrs_2, int RB)
{
	int SC = Num_SC_RB * RB;
	dmrs_1[0:SC] = gride_in[3][0:SC]; //dmrs_1
	dmrs_2[0:SC] = gride_in[10][0:SC];//dmrs_2
	/*Serial output data*/
	for (int i = 0; i < NUM_SYM_SUBFRAME ; i++)
	{
		switch (i)
		{
		case 0: case 1: case 2:
			data[i*SC:SC] = gride_in[i][0:SC]; 
			break;
		case 3: case 10:
			break;
		default:
			if (i>3 && i<10)
				data[(i - 1)*SC:SC] = gride_in[i][0:SC];
			else
				data[(i - 2)*SC:SC] = gride_in[i][0:SC];
		}
	}
}