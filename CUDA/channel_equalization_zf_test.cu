


#include "channel_equalization_zf.cuh"





int main(int argc, char **argv) {


	const int M_pusch_rb = 100;
	const int M_pusch_sc = N_sc_rb * M_pusch_rb;

	//input
	cufftComplex* subframe = (cufftComplex *)malloc(sizeof(cufftComplex)*M_pusch_sc*N_data_symbs_per_subframe);
	cufftComplex* channel = (cufftComplex *)malloc(sizeof(cufftComplex)*M_pusch_sc);


	for (int i = 0; i < M_pusch_sc; i++)
	{
		channel[i].x = i;
		channel[i].y = i+1;
	}

	for (int i = 0; i < (M_pusch_sc*N_data_symbs_per_subframe); i++)
	{
		subframe[i].x = i+3;
		subframe[i].y = i+2;
	}

	//For output
	cufftComplex *equalized_symbols;



	channel_equalization_zf(subframe, M_pusch_sc, channel, &equalized_symbols);

	//Print results
	for (int i = 0; i < (M_pusch_sc*N_data_symbs_per_subframe); i++)
	{
		printf("index %i = %f + %f i\n", i + 1, equalized_symbols[i].x, equalized_symbols[i].y);
	}

	FILE * pFile;

	pFile = fopen("results.m", "w");
	fprintf(pFile, "x = [ ");
	for (int i = 0; i < (M_pusch_sc*N_data_symbs_per_subframe); i++)
	{
		fprintf(pFile, "%f ", equalized_symbols[i].x);
	}
	fprintf(pFile, " ]; ");
	fprintf(pFile, "\n");
	fprintf(pFile, "y = [ ");
	for (int i = 0; i < (M_pusch_sc*N_data_symbs_per_subframe); i++)
	{
		fprintf(pFile, "%f ", equalized_symbols[i].y);
	}
	fprintf(pFile, " ]; ");
	fclose(pFile);
	return 0;

}
