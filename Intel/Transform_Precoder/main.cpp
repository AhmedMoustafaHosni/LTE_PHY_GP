/* This app is for test the transform precoding using unit steps the output must be [1200 0 0 0 ....0] for each 1200 element*/
// if you want to see the output goto Transform_precoding function and decomment the verfier 
// By: Khaled Ahmed Ali 

#include "Intel_Header.h"

int main()
{
	// test with step functions
	MKL_Complex8* x = 0;

	//
	int SC = M_PUSCH_SC;
	
	/*Allocate input array*/
	x = (MKL_Complex8*)mkl_malloc(DATA_SIZE*sizeof(MKL_Complex8), 64);
	if (0 == x)
	{
		printf("Error not enough memory in heap\n");
		return 0;
	}

	//initialize the steps 
	for (int i = 0; i < DATA_SIZE; i++)
	{
		x[i].real = 1;
		x[i].imag = 0;
	}

	if (Transform_precoder(x, SC) != 0)
	{
		printf("Error in Transform Precoding\n");
		return 0;
	}

	/*Free data array*/
	mkl_free(x);

	return 0;
}