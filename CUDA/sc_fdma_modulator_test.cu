/*
% Function:	sc_fdma_modulator
% Description:	Generates sc-fdma signal of the subframe
% Inputs:	*subframe_h:	received DMRS number 1
% 		M_pusch_rb	numer of resource blocks assigned to ue
% Outputs:	*pusch_bb_h	base band signal
By: Mohammed Mostafa
*/

#include "sc_fdma_modulator.cuh"

int main(int argc, char **argv) {
							
	const int M_pusch_rb = 100;
	const int M_pusch_sc = N_sc_rb * M_pusch_rb; 

	//input
	cufftComplex* subframe_h = (cufftComplex *)malloc(sizeof(cufftComplex)*N_symbs_per_subframe*M_pusch_sc);

	int j = 1;
	for (int i = 0; i < N_symbs_per_subframe*M_pusch_sc ; i++)
	{
			subframe_h[i].x = rand()/(float)RAND_MAX*10;
			subframe_h[i].y = rand()/(float)RAND_MAX*10;
			j++;
			if (j == 1201)
				j = 1;
	}

	//For output
	cufftComplex *pusch_bb_h;

	//Call the Transform Precoder Function
	sc_fdma_modulator(subframe_h, M_pusch_rb, &pusch_bb_h);

	//Print results
	for (int i = 0; i < modulated_subframe_length; i++)
	{
			printf("idx = %d \t %f \t %f \n", i + 1, pusch_bb_h[i].x, pusch_bb_h[i].y);
	}
	
	//To compare with MATLAB results
	FILE *results;
	if ((results = freopen("TP_Results.m", "w+", stdout)) == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}

	printf("clear; clc;\nsymbols_real = [ ");
	for (int i = 0; i < (modulated_subframe_length); i++)
	{
		printf("%10f", pusch_bb_h[i].x);
		if (i != ((modulated_subframe_length)-1))
			printf(",");
	}

	printf(" ];\nsymbols_imag = [ ");

	for (int i = 0; i < (modulated_subframe_length); i++)
	{
		printf("%10f", pusch_bb_h[i].y);
		if (i != ((modulated_subframe_length)-1))
			printf(",");
	}

	printf(" ];\n");
	printf("symbols_CUDA = symbols_real + 1i * symbols_imag;\n");

	fclose(results);

	FILE *results1;
	if ((results1 = freopen("TP_input_.m", "w+", stdout)) == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}

	printf("clear; clc;\nsymbols_in_real = [ ");
	for (int i = 0; i < (16800); i++)
	{
		printf("%10f", subframe_h[i].x);
		if (i != ((16800)-1))
			printf(",");
	}

	printf(" ];\nsymbols_in_imag = [ ");

	for (int i = 0; i < (16800); i++)
	{
		printf("%10f", subframe_h[i].y);
		if (i != ((16800)-1))
			printf(",");
	}

	printf(" ];\n");
	printf("symbols_input_CUDA = symbols_in_real + 1i * symbols_in_imag;\n");
	fclose(results1);

	return 0;

}
