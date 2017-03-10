

#include "channel_estimation.cuh"





int main(int argc, char **argv) {


	const int M_pusch_rb = 100;
	const int M_pusch_sc = N_sc_rb * M_pusch_rb;

	//input
	cufftComplex* dmrs_0 = (cufftComplex *)malloc(sizeof(cufftComplex)*M_pusch_sc);
	cufftComplex* dmrs_1 = (cufftComplex *)malloc(sizeof(cufftComplex)*M_pusch_sc);
	cufftComplex* symb_0 = (cufftComplex *)malloc(sizeof(cufftComplex)*M_pusch_sc);
	cufftComplex* symb_1 = (cufftComplex *)malloc(sizeof(cufftComplex)*M_pusch_sc);

	for (int i = 0; i < M_pusch_sc; i++)
	{
		dmrs_0[i].x = i;
		dmrs_1[i].x = i+5;
		symb_0[i].x = i+2;
		symb_1[i].x = i+10;
		dmrs_0[i].y = i+1;
		dmrs_1[i].y = i+7;
		symb_0[i].y = i+4;
		symb_1[i].y = i+3;
	}

	//For output
	cufftComplex *channel;


	//Call the Transform Precoder Function
	channe_estimation(symb_0, symb_1, dmrs_0, dmrs_1, M_pusch_sc, &channel);

	//Print results

	for (int i = 0; i < M_pusch_sc; i++)
	{
		printf("index %i = %f + %f i\n",i+1, channel[i].x, channel[i].y);
	}

	FILE * pFile;

	pFile = fopen("results.m", "w");
	fprintf(pFile, "x = [ ");
	for (int i = 0; i <  M_pusch_sc; i++)
	{
		fprintf(pFile, "%f ", channel[i].x);
	}
	fprintf(pFile, " ]; ");
	fprintf(pFile, "\n");
	fprintf(pFile, "y = [ ");
	for (int i = 0; i < M_pusch_sc; i++)
	{
		fprintf(pFile, "%f ", channel[i].y);
	}
	fprintf(pFile, " ]; ");
	fclose(pFile);
	return 0;

}
