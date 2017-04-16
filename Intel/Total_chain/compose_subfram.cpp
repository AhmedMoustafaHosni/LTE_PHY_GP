/*******************************************************************************
* Function:    compose_subframe
* Description: gride mapping
* Inputs:      data      - corresponding to data symbols from transform precoder
*			   dmrs1     - Demodulated referance  signal 1 
*              dmrs1     - Demodulated referance  signal 2 
*			   RB        - Number of RBs assigned to the ue
*
* Outputs:     gride     - Pointer to Pointer to the gride 
*
*
* by: Khaled Ahmed Ali
********************************************************************************/


#include "Intel_siso.h"

MKL_Complex8 ** compose_subframe(MKL_Complex8* data, MKL_Complex8* dmrs_1, MKL_Complex8* dmrs_2,int RB)
{
	int SC = RB *Num_SC_RB; // 100 * 12 
	MKL_Complex8 ** gride = (MKL_Complex8 **)malloc(NUM_SYM_SUBFRAME * sizeof(MKL_Complex8 *));
	for (int i = 0; i < NUM_SYM_SUBFRAME; i++)
	{
		gride[i] = (MKL_Complex8 *) malloc(SC * sizeof(MKL_Complex8));
		switch (i)
		{
		case 0: case 1:case 2:
			gride[i][0:SC] = data[i*SC:SC];
			break;
		case 3:
			gride[i][0:SC] = dmrs_1[0:SC];
			break;
		case 10:
			gride[i][0:SC] = dmrs_2[0:SC];
			break;
		default:
			if (i < 10)
			gride[i][0:SC] = data[(i-1)*SC:SC];
			else
			gride[i][0:SC] = data[(i - 2)*SC:SC];
		}
	}
	return gride;
}