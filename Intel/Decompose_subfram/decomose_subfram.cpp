#include "Header.h"

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
		case 3: case 10:
			break;
		default:
			data[i*SC:SC] = gride_in[i][0:SC];
		}
	}
}