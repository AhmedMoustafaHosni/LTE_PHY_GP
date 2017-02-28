/* This app is for test the transform pr-decoding using input data of [sqrt(M_PUSCH_SC) . . . . sqrt(M_PUSCH_SC)] the output must be [1 0 0 0 0 . . . .  0] for each M_PUSCH_SC (1200 in our case)*/
// if you want to see the output goto Transform_predecoding function and decomment the verfier 
// By: Khaled Ahmed Ali 

#include "Intel_Header.h"

int main()
{
	// test with [sqrt(M_PUSCH_SC) . . . . . sqrt(M_PUSCH_SC)]
	MKL_Complex8* x = 0;

	// Subcarriers assigned for the UE
	int SC = M_PUSCH_SC;
	
	/*Allocate input array*/
	x = (MKL_Complex8*)mkl_malloc(DATA_SIZE*sizeof(MKL_Complex8), 64);
	if (0 == x)
	{
		printf("Error not enough memory in heap\n");
		return 0;
	}

	//initialize the input Data 
	for (int i = 0; i < DATA_SIZE; i++)
	{
			x[i].real = sqrt((float)SC);
			x[i].imag = 0;
	}

	//Calling the tranform predecoder
	if (Transform_predecoder(x, SC) != 0)
	{
		printf("Error in Transform Precoding\n");
		return 0;
	}

	/*Free data array*/
	mkl_free(x);

	return 0;
}