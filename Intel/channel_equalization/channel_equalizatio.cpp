
/*

Function:		 channel_equalization_zf
Description : equalise the channel effecct on the received signal


Inputs :	  modulated_subframe - received subframe without demodulation signal in one vector
			  channel - estimated channel


Outputs :    equalised symbols

// By : Ahmed Moustafa

*/


#include "channel_equalizatio.h"





void channel_equalization(MKL_Complex8* modulated_subframe, MKL_Complex8* channel , MKL_Complex8* equalized_subframe, const int M_pusch_sc)
{

	for (int j = 0; j < N_data_symbs_per_subframe; j++)
	for (int i = 0; i < M_pusch_sc; i++)
	{
		equalized_subframe[j*M_pusch_sc+i].real = (modulated_subframe[j*M_pusch_sc + i].real*channel[i].real + modulated_subframe[j*M_pusch_sc + i].imag*channel[i].imag) / (channel[i].real*channel[i].real + channel[i].imag*channel[i].imag);
		equalized_subframe[j*M_pusch_sc + i].imag = (modulated_subframe[j*M_pusch_sc + i].imag*channel[i].real - modulated_subframe[j*M_pusch_sc + i].real*channel[i].imag) / (channel[i].real*channel[i].real + channel[i].imag*channel[i].imag);
	}

}