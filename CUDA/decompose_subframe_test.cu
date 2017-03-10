/*
% Function:		decompose_subframe
% Description:	compose the subframe by multiplexing the dmrs signal and data
% Inputs:		*subframe_h			the subframe with data of all ofdm symbols
%				M_pusch_rb			number of resource blocks assigned to the ue
% Outputs:		*complex_data_h:	complex data to be sent in subframe
%				*dmrs_1_h:			demodulation reference signal number 1
%				*dmrs_2_h:			demodulation reference signal number 2
By: Ahmad Nour & Mohammed Mostafa
*/

#include "decompose_subframe.cuh"


int main(int argc, char **argv) {

	const int M_pusch_rb = 100;
	const int M_pusch_sc = N_sc_rb * M_pusch_rb;

	//input
	cufftComplex* subframe_h = (cufftComplex *)malloc(sizeof(cufftComplex) * N_symbs_per_subframe * M_pusch_sc);


	for (int i = 0; i < N_symbs_per_subframe * M_pusch_sc; i++)
	{
		subframe_h[i].x = rand() / (float)RAND_MAX * 10;
		subframe_h[i].y = rand() / (float)RAND_MAX * 10;
	}


	//For output
	cufftComplex *complex_data_h;
	cufftComplex *dmrs_1_h;
	cufftComplex *dmrs_2_h;

	//Call the Transform Precoder Function
	decompose_subframe(subframe_h, M_pusch_rb, &complex_data_h, &dmrs_1_h, &dmrs_2_h);

	//Print results
	for (int i = 0; i < 12*M_pusch_sc; i++)
	{
		printf("idx = %d \t %f \t %f \n", i + 1, complex_data_h[i].x, complex_data_h[i].y);
	}

	//input file
	FILE *results;
	if ((results = freopen("decompose_subframe_input.m", "w+", stdout)) == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}

	printf("clear; clc;\nsymbols_real = [ ");
	for (int i = 0; i < (N_symbs_per_subframe*M_pusch_sc); i++)
	{
		printf("%10f", subframe_h[i].x);
		if (i != ((N_symbs_per_subframe*M_pusch_sc) - 1))
			printf(",");
	}

	printf(" ];\nsymbols_imag = [ ");

	for (int i = 0; i < (N_symbs_per_subframe*M_pusch_sc); i++)
	{
		printf("%10f", subframe_h[i].y);
		if (i != ((N_symbs_per_subframe*M_pusch_sc) - 1))
			printf(",");
	}

	printf(" ];\n");
	printf("subframe_CUDA = symbols_real + 1i * symbols_imag;\n");
	fclose(results);

	//output file
	FILE *results1;
	if ((results1 = freopen("decompose_subframe_Results_.m", "w+", stdout)) == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}

	//complex_data
	printf("clear; clc;\ncomplex_data_real = [ ");
	for (int i = 0; i < (12 * M_pusch_sc); i++)
	{
		printf("%10f", complex_data_h[i].x);
		if (i != ((12 * M_pusch_sc) - 1))
			printf(",");
	}

	printf(" ];\ncomplex_data_imag = [ ");

	for (int i = 0; i < (12 * M_pusch_sc); i++)
	{
		printf("%10f", complex_data_h[i].y);
		if (i != ((12 * M_pusch_sc) - 1))
			printf(",");
	}

	printf(" ];\n");
	printf("complex_data_CUDA = complex_data_real + 1i * complex_data_imag;\n");

	//dmrs_1
	printf("dmrs1_real = [ ");
	for (int i = 0; i < (M_pusch_sc); i++)
	{
		printf("%10f", dmrs_1_h[i].x);
		if (i != ((M_pusch_sc)-1))
			printf(",");
	}

	printf(" ];\ndmrs1_imag = [ ");

	for (int i = 0; i < (M_pusch_sc); i++)
	{
		printf("%10f", dmrs_1_h[i].y);
		if (i != ((M_pusch_sc)-1))
			printf(",");
	}

	printf(" ];\n");
	printf("dmrs1_CUDA = dmrs1_real + 1i * dmrs1_imag;\n");

	//dmrs2
	printf("dmrs2_real = [ ");
	for (int i = 0; i < (M_pusch_sc); i++)
	{
		printf("%10f", dmrs_2_h[i].x);
		if (i != ((M_pusch_sc)-1))
			printf(",");
	}

	printf(" ];\ndmrs2_imag = [ ");

	for (int i = 0; i < (M_pusch_sc); i++)
	{
		printf("%10f", dmrs_2_h[i].y);
		if (i != ((M_pusch_sc)-1))
			printf(",");
	}

	printf(" ];\n");
	printf("dmrs2_CUDA = dmrs2_real + 1i * dmrs2_imag;\n");

	//close input file
	fclose(results1);

	return 0;

}