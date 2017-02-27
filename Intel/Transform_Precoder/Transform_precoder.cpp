#include "Intel_Header.h"

MKL_LONG Transform_precoder(MKL_Complex8* x, int N)
{

//----------------//
//Initializations//
//---------------//

	/* Temporary variable used for division*/
	MKL_Complex8 *temp = 0; 

	/* Scaling DFT*/
	float Scale = 1/sqrtf(N);

	/* Execution status */
	MKL_LONG status = 0;

	DFTI_DESCRIPTOR_HANDLE hand = 0;

//----------------//
//  FFT 5 STEPS   //
//---------------//

	/*create descriptor*/
	status = DftiCreateDescriptor(&hand, DFTI_SINGLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
	if (0 != status) goto failed;

	/* Scaling */
	status = DftiSetValue(hand, DFTI_FORWARD_SCALE,Scale);
	if (0 != status) goto failed;

	/*Commiting descriptor*/
	status = DftiCommitDescriptor(hand);
	if (0 != status) goto failed;

	/*Allocate temporary memory for one single subset in fft*/
	temp = (MKL_Complex8*)mkl_malloc(N*sizeof(MKL_Complex8), 64);

	/*Time Capture*/
	double s_initial = 0, s_elapsed = 0;
	s_initial = dsecnd();  // sample time

	/*Initialize input for forward transform*/ //just for test
	for (int i = 0; i < DATA_SIZE / N; i++)
	{
		/*Dividing the input into subsets in temp*/
		for (int j = 0; j < N; j++)
			temp[j] = x[j + i * N];
		/*Compute forward transform*/
		status = DftiComputeForward(hand, temp);
		if (0 != status) goto failed;
		/*The divided part in the Output Back to X*/
		for (int j = 0; j < N; j++)
			x[j + i * N] = temp[j];
	}
	s_elapsed = (dsecnd() - s_initial);
	printf(" completed with == \n == at %.5f milliseconds == \n\n", (s_elapsed * 1000));

	/*Verify the result of forward FFT if all are step functions*/
	/*
	for (int i = 0; i < DATA_SIZE; i++)
	{
		if (i % N == 0) //must be all 1200
			printf("REAL%d = %f , IMAG%d = %f\n", i + 1, x[i].real, i + 1, x[i].imag);
	}
	*/
END:

	/*Release the DFTI descriptor*/
	DftiFreeDescriptor(&hand);
	return status;
failed:
	status = 1;
	goto END;
}