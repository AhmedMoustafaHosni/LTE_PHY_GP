

#include "channel_estimation.h"





int main(int argc, char **argv) {


	const int M_pusch_rb = 100;
	const int M_pusch_sc = N_sc_rb * M_pusch_rb;


	//For output
	MKL_Complex8* channel = (MKL_Complex8 *)malloc(sizeof(MKL_Complex8)*M_pusch_sc);


	//input
	MKL_Complex8* dmrs_0 = (MKL_Complex8 *)malloc(sizeof(MKL_Complex8)*M_pusch_sc);
	MKL_Complex8* dmrs_1 = (MKL_Complex8 *)malloc(sizeof(MKL_Complex8)*M_pusch_sc);
	MKL_Complex8* symb_0 = (MKL_Complex8 *)malloc(sizeof(MKL_Complex8)*M_pusch_sc);
	MKL_Complex8* symb_1 = (MKL_Complex8 *)malloc(sizeof(MKL_Complex8)*M_pusch_sc);

	for (int i = 0; i < M_pusch_sc; i++)
	{
		dmrs_0[i].real = i;
		dmrs_1[i].real = i + 5;
		symb_0[i].real = i + 2;
		symb_1[i].real = i + 10;
		dmrs_0[i].imag = i + 1;
		dmrs_1[i].imag = i + 7;
		symb_0[i].imag = i + 4;
		symb_1[i].imag = i + 3;
	}

	


	//Call the Transform Precoder Function
	double time_before = dsecnd();
	channe_estimation(symb_0, symb_1, dmrs_0, dmrs_1, M_pusch_sc, channel);
	double execution_time = (dsecnd() - time_before);

	//print_array_of_float(bits, number_of_bits);
	printf("\n");
	printf("time = %f   milliseconds", execution_time*1000.0);
	printf("\n");

	 time_before = dsecnd();
	channe_estimation(symb_0, symb_1, dmrs_0, dmrs_1, M_pusch_sc, channel);
	 execution_time = (dsecnd() - time_before);

	//Print results

	for (int i = 0; i < M_pusch_sc; i++)
	{
		printf("index %i = %f + %f i\n", i + 1, channel[i].real, channel[i].imag);
	}

	//print_array_of_float(bits, number_of_bits);
	printf("\n");
	printf("time = %f   milliseconds", execution_time*1000.0);
	printf("\n");

	ofstream  pFile; 

	pFile.open("results.m"); 

	pFile << "x = [ ";
	for (int i = 0; i < M_pusch_sc; i++)
	{
		pFile << channel[i].real;
		pFile << ' ';
	}
	pFile <<  " ]; ";
	pFile << "\n";
	pFile << "y = [ ";
	for (int i = 0; i < M_pusch_sc; i++)
	{
		pFile << channel[i].imag;
		pFile << ' ';
	}
	pFile << " ]; ";

	pFile.close();
	return 0;

}
