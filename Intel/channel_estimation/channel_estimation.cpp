//  Function:    channe_estimation
//	Description : Generates channel estimation
//	Inputs : symb_0 - received DMRS number 1
//			 symb_1 - received DMRS number 2
//			 dmrs_0 - generated DMRS number 1
//			 dmrs_1 - generated DMRS number 2
//			 M_pusch_sc - number of subcarriers allocated to ue
//	Outputs : channel - channel estimation matrix to be used for equalization
//	By : Ahmed Moustafa

#include "channel_estimation.h"

void divide_complex_vector(MKL_Complex8* a, MKL_Complex8* b, MKL_Complex8* c, const int n);
void time_filter(MKL_Complex8* a, MKL_Complex8* b, MKL_Complex8* c, const int n);


void channe_estimation(MKL_Complex8* symb_0, MKL_Complex8* symb_1, MKL_Complex8* dmrs_0, MKL_Complex8* dmrs_1, const int M_pusch_sc, MKL_Complex8* channel)
{
	MKL_Complex8* H_1;
	MKL_Complex8* H_2;
	MKL_Complex8* time_filtered;


	H_1 = (MKL_Complex8*)mkl_malloc(sizeof(MKL_Complex8)*M_pusch_sc, 32);
	H_2 = (MKL_Complex8*)mkl_malloc(sizeof(MKL_Complex8)*M_pusch_sc, 32);
	time_filtered = (MKL_Complex8*)mkl_malloc(sizeof(MKL_Complex8)*M_pusch_sc, 32);

	//vcDiv(M_pusch_sc, symb_0, dmrs_0, H_1);
	//vcDiv(M_pusch_sc, symb_1, dmrs_1, H_2);

	divide_complex_vector(symb_0, dmrs_0, H_1, M_pusch_sc);
	divide_complex_vector(symb_1, dmrs_1, H_2, M_pusch_sc);

	//vcAdd(M_pusch_sc, H_1, H_2, time_filtered);
	//cblas_csscal(M_pusch_sc, (float)0.5, time_filtered, 1);

	time_filter(H_1, H_2, time_filtered, M_pusch_sc);

	for (int i = 0; i < M_pusch_sc; i++)
	{
		if (i > 8 && i < (M_pusch_sc - 9))
		{
			channel[i].real = (time_filtered[i - 9].real + time_filtered[i - 8].real + time_filtered[i - 7].real + time_filtered[i - 6].real + time_filtered[i - 5].real + time_filtered[i - 4].real + time_filtered[i - 3].real + time_filtered[i - 2].real + time_filtered[i - 1].real + time_filtered[i].real + time_filtered[i + 1].real + time_filtered[i + 2].real + time_filtered[i + 3].real + time_filtered[i + 4].real + time_filtered[i + 5].real + time_filtered[i + 6].real + time_filtered[i + 7].real + time_filtered[i + 8].real + time_filtered[i + 9].real) / 19;
			channel[i].imag = (time_filtered[i - 9].imag + time_filtered[i - 8].imag + time_filtered[i - 7].imag + time_filtered[i - 6].imag + time_filtered[i - 5].imag + time_filtered[i - 4].imag + time_filtered[i - 3].imag + time_filtered[i - 2].imag + time_filtered[i - 1].imag + time_filtered[i].imag + time_filtered[i + 1].imag + time_filtered[i + 2].imag + time_filtered[i + 3].imag + time_filtered[i + 4].imag + time_filtered[i + 5].imag + time_filtered[i + 6].imag + time_filtered[i + 7].imag + time_filtered[i + 8].imag + time_filtered[i + 9].imag) / 19;
		}
		else if (i == 8 || i == (M_pusch_sc - 9))
		{
			channel[i].real = (time_filtered[i - 8].real + time_filtered[i - 7].real + time_filtered[i - 6].real + time_filtered[i - 5].real + time_filtered[i - 4].real + time_filtered[i - 3].real + time_filtered[i - 2].real + time_filtered[i - 1].real + time_filtered[i].real + time_filtered[i + 1].real + time_filtered[i + 2].real + time_filtered[i + 3].real + time_filtered[i + 4].real + time_filtered[i + 5].real + time_filtered[i + 6].real + time_filtered[i + 7].real + time_filtered[i + 8].real) / 17;
			channel[i].imag = (time_filtered[i - 8].imag + time_filtered[i - 7].imag + time_filtered[i - 6].imag + time_filtered[i - 5].imag + time_filtered[i - 4].imag + time_filtered[i - 3].imag + time_filtered[i - 2].imag + time_filtered[i - 1].imag + time_filtered[i].imag + time_filtered[i + 1].imag + time_filtered[i + 2].imag + time_filtered[i + 3].imag + time_filtered[i + 4].imag + time_filtered[i + 5].imag + time_filtered[i + 6].imag + time_filtered[i + 7].imag + time_filtered[i + 8].imag) / 17;
		}
		else if (i == 7 || i == (M_pusch_sc - 8))
		{
			channel[i].real = (time_filtered[i - 7].real + time_filtered[i - 6].real + time_filtered[i - 5].real + time_filtered[i - 4].real + time_filtered[i - 3].real + time_filtered[i - 2].real + time_filtered[i - 1].real + time_filtered[i].real + time_filtered[i + 1].real + time_filtered[i + 2].real + time_filtered[i + 3].real + time_filtered[i + 4].real + time_filtered[i + 5].real + time_filtered[i + 6].real + time_filtered[i + 7].real) / 15;
			channel[i].imag = (time_filtered[i - 7].imag + time_filtered[i - 6].imag + time_filtered[i - 5].imag + time_filtered[i - 4].imag + time_filtered[i - 3].imag + time_filtered[i - 2].imag + time_filtered[i - 1].imag + time_filtered[i].imag + time_filtered[i + 1].imag + time_filtered[i + 2].imag + time_filtered[i + 3].imag + time_filtered[i + 4].imag + time_filtered[i + 5].imag + time_filtered[i + 6].imag + time_filtered[i + 7].imag) / 15;
		}
		else if (i == 6 || i == (M_pusch_sc - 7))
		{
			channel[i].real = (time_filtered[i - 6].real + time_filtered[i - 5].real + time_filtered[i - 4].real + time_filtered[i - 3].real + time_filtered[i - 2].real + time_filtered[i - 1].real + time_filtered[i].real + time_filtered[i + 1].real + time_filtered[i + 2].real + time_filtered[i + 3].real + time_filtered[i + 4].real + time_filtered[i + 5].real + time_filtered[i + 6].real) / 13;
			channel[i].imag = (time_filtered[i - 6].imag + time_filtered[i - 5].imag + time_filtered[i - 4].imag + time_filtered[i - 3].imag + time_filtered[i - 2].imag + time_filtered[i - 1].imag + time_filtered[i].imag + time_filtered[i + 1].imag + time_filtered[i + 2].imag + time_filtered[i + 3].imag + time_filtered[i + 4].imag + time_filtered[i + 5].imag + time_filtered[i + 6].imag) / 13;
		}
		else if (i == 5 || i == (M_pusch_sc - 6))
		{
			channel[i].real = (time_filtered[i - 5].real + time_filtered[i - 4].real + time_filtered[i - 3].real + time_filtered[i - 2].real + time_filtered[i - 1].real + time_filtered[i].real + time_filtered[i + 1].real + time_filtered[i + 2].real + time_filtered[i + 3].real + time_filtered[i + 4].real + time_filtered[i + 5].real) / 11;
			channel[i].imag = (time_filtered[i - 5].imag + time_filtered[i - 4].imag + time_filtered[i - 3].imag + time_filtered[i - 2].imag + time_filtered[i - 1].imag + time_filtered[i].imag + time_filtered[i + 1].imag + time_filtered[i + 2].imag + time_filtered[i + 3].imag + time_filtered[i + 4].imag + time_filtered[i + 5].imag) / 11;
		}
		else if (i == 4 || i == (M_pusch_sc - 5))
		{
			channel[i].real = (time_filtered[i - 4].real + time_filtered[i - 3].real + time_filtered[i - 2].real + time_filtered[i - 1].real + time_filtered[i].real + time_filtered[i + 1].real + time_filtered[i + 2].real + time_filtered[i + 3].real + time_filtered[i + 4].real) / 9;
			channel[i].imag = (time_filtered[i - 4].imag + time_filtered[i - 3].imag + time_filtered[i - 2].imag + time_filtered[i - 1].imag + time_filtered[i].imag + time_filtered[i + 1].imag + time_filtered[i + 2].imag + time_filtered[i + 3].imag + time_filtered[i + 4].imag) / 9;
		}
		else if (i == 3 || i == (M_pusch_sc - 4))
		{
			channel[i].real = (time_filtered[i - 3].real + time_filtered[i - 2].real + time_filtered[i - 1].real + time_filtered[i].real + time_filtered[i + 1].real + time_filtered[i + 2].real + time_filtered[i + 3].real) / 7;
			channel[i].imag = (time_filtered[i - 3].imag + time_filtered[i - 2].imag + time_filtered[i - 1].imag + time_filtered[i].imag + time_filtered[i + 1].imag + time_filtered[i + 2].imag + time_filtered[i + 3].imag) / 7;
		}
		else if (i == 2 || i == (M_pusch_sc - 3))
		{
			channel[i].real = (time_filtered[i - 2].real + time_filtered[i - 1].real + time_filtered[i].real + time_filtered[i + 1].real + time_filtered[i + 2].real) / 5;
			channel[i].imag = (time_filtered[i - 2].imag + time_filtered[i - 1].imag + time_filtered[i].imag + time_filtered[i + 1].imag + time_filtered[i + 2].imag) / 5;
		}
		else if (i == 1 || i == (M_pusch_sc - 2))
		{
			channel[i].real = (time_filtered[i - 1].real + time_filtered[i].real + time_filtered[i + 1].real) / 3;
			channel[i].imag = (time_filtered[i - 1].imag + time_filtered[i].imag + time_filtered[i + 1].imag) / 3;
		}
		else
		{
			channel[i].real = time_filtered[i].real;
			channel[i].imag = time_filtered[i].imag;
		}
	}
	
	free(time_filtered);
	free(H_2);
	free(H_1);
}



void divide_complex_vector(MKL_Complex8* a, MKL_Complex8* b, MKL_Complex8* c, const int n)
{
	for (int i = 0; i < n; i++)
	{
		c[i].real = (a[i].real*b[i].real + a[i].imag*b[i].imag) / (b[i].real*b[i].real + b[i].imag*b[i].imag);
		c[i].imag = (a[i].imag*b[i].real - a[i].real*b[i].imag) / (b[i].real*b[i].real + b[i].imag*b[i].imag);
	}
}

void time_filter(MKL_Complex8* a, MKL_Complex8* b, MKL_Complex8* c, const int n)
{
	for (int i = 0; i < n; i++)
	{
		c[i].real = (a[i].real + b[i].real) / 2;
		c[i].imag = (a[i].imag + b[i].imag) / 2;
	}
}
