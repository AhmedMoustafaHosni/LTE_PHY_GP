/*
% Function:		sc_fdma_demodulator
% Description:	Generates complex symbols from the sc-fdma symbols
% Inputs		*pusch_bb_h		sc-fdma symbols
				M_pusch_rb		numer of resource blocks assigned to ue
% Outputs:		*symbs_h		output symbols 
By: Mohammed Mostafa
*/

#include "sc_fdma_demodulator.cuh"

int main(int argc, char **argv) {
	
	const int M_pusch_rb = 100;
	const int M_pusch_sc = N_sc_rb * M_pusch_rb; 
	//input
	cufftComplex* pusch_bb_h = (cufftComplex *)malloc(sizeof(cufftComplex)*modulated_subframe_length);
	
	for (int i = 0; i < modulated_subframe_length ; i++)
	{
		pusch_bb_h[i].x = rand()/(float)RAND_MAX*10;
		pusch_bb_h[i].y = rand()/(float)RAND_MAX*10;
	}

	//For output
	cufftComplex *symbs_h;

	//Call the Transform Precoder Function
	sc_fdma_demodulator(pusch_bb_h, M_pusch_rb, &symbs_h);

	//Print results
	for (int i = 0; i < 14*M_pusch_sc ; i++)
	{
			printf("idx = %d \t %f \t %f \n", i + 1, symbs_h[i].x, symbs_h[i].y);
	}
	
	//To compare with MATLAB results
	//Run the file (Demapper_Results.m)
	FILE *results;
	if ((results = freopen("TP_Results.m", "w+", stdout)) == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}

	//output file
	printf("clear; clc;\nsymbols_real = [ ");
	for (int i = 0; i < (14*M_pusch_sc); i++)
	{
		printf("%10f", symbs_h[i].x);
		if (i != ((14 * M_pusch_sc)-1))
			printf(",");
	}

	printf(" ];\nsymbols_imag = [ ");

	for (int i = 0; i < (14 * M_pusch_sc); i++)
	{
		printf("%10f", symbs_h[i].y);
		if (i != ((14 * M_pusch_sc)-1))
			printf(",");
	}

	printf(" ];\n");
	printf("symbols_CUDA = symbols_real + 1i * symbols_imag;\n");
	fclose(results);

	//input file
	FILE *results1;
	if ((results1 = freopen("TP_input_.m", "w+", stdout)) == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}

	printf("clear; clc;\nsymbols_in_real = [ ");
	for (int i = 0; i < (modulated_subframe_length); i++)
	{
		printf("%10f", pusch_bb_h[i].x);
		if (i != ((modulated_subframe_length)-1))
			printf(",");
	}

	printf(" ];\nsymbols_in_imag = [ ");

	for (int i = 0; i < (modulated_subframe_length); i++)
	{
		printf("%10f", pusch_bb_h[i].y);
		if (i != ((modulated_subframe_length)-1))
			printf(",");
	}

	printf(" ];\n");
	printf("symbols_input_CUDA = symbols_in_real + 1i * symbols_in_imag;\n");
	fclose(results1);

	return 0;

}
