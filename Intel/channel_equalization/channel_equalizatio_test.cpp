

#include "channel_equalizatio.h"




int main(int argc, char **argv) {


	const int M_pusch_rb = 100;
	const int M_pusch_sc = N_sc_rb * M_pusch_rb;

	//input
	MKL_Complex8* subframe = (MKL_Complex8 *)malloc(sizeof(MKL_Complex8)*M_pusch_sc*N_data_symbs_per_subframe);
	MKL_Complex8* channel = (MKL_Complex8 *)malloc(sizeof(MKL_Complex8)*M_pusch_sc);


	for (int i = 0; i < M_pusch_sc; i++)
	{
		channel[i].real = i;
		channel[i].imag = i + 1;
	}

	for (int i = 0; i < (M_pusch_sc*N_data_symbs_per_subframe); i++)
	{
		subframe[i].real = i + 3;
		subframe[i].imag = i + 2;
	}

	//For output
	MKL_Complex8 *equalized_symbols = (MKL_Complex8 *)malloc(sizeof(MKL_Complex8)*M_pusch_sc*N_data_symbs_per_subframe);


	double time_before = dsecnd();
	channel_equalization(subframe, channel, equalized_symbols, M_pusch_sc);
	double execution_time = (dsecnd() - time_before);


	//print_array_of_float(bits, number_of_bits);
	printf("\n");
	printf("time = %f   milliseconds", execution_time*1000.0);
	printf("\n");


	 time_before = dsecnd();
	channel_equalization(subframe, channel, equalized_symbols, M_pusch_sc);
	 execution_time = (dsecnd() - time_before);


	//Print results
	for (int i = 0; i < (M_pusch_sc*N_data_symbs_per_subframe); i++)
	{
		printf("index %i = %f + %f i\n", i + 1, equalized_symbols[i].real, equalized_symbols[i].imag);
	}


	//print_array_of_float(bits, number_of_bits);
	printf("\n");
	printf("time = %f   milliseconds", execution_time*1000.0);
	printf("\n");

	ofstream  pFile;

	pFile.open("results.m");

	pFile << "x = [ ";
	for (int i = 0; i < M_pusch_sc*N_data_symbs_per_subframe; i++)
	{
		pFile << equalized_symbols[i].real;
		pFile << ' ';
	}
	pFile << " ]; ";
	pFile << "\n";
	pFile << "y = [ ";
	for (int i = 0; i < M_pusch_sc*N_data_symbs_per_subframe; i++)
	{
		pFile << equalized_symbols[i].imag;
		pFile << ' ';
	}
	pFile << " ]; ";

	pFile.close();
	return 0;

}
