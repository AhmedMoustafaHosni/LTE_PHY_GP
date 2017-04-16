/*******************************************************************************
* Function:    Transform_predecoder
* Description: predecoding the data symbols on subsets of the available subcarriers
* Inputs:      x          - Precodded data symbols
*			   N          - Number of subcarriers assigned to the ue
* Outputs:     x		  - Data symbols for demapper
*
* by: Khaled Ahmed Ali
********************************************************************************/

#include "Intel_siso.h"

MKL_LONG Transform_predecoder(MKL_Complex8* x, int N)
{

//----------------//
//Initializations//
//---------------//

	/* Temporary variable used for division*/
	MKL_Complex8 *temp = 0; 

	/* Scaling IDFT*/ // note that the scaling here must take this formula [(1/N) for ifft formula + Transform predecoder scale(1/sqrt)]
	float Scale = 1/sqrt(N);

	/* Execution status */
	MKL_LONG status = 0;

	DFTI_DESCRIPTOR_HANDLE hand = 0;

//----------------//
//  IFFT 5 STEPS   //
//---------------//

	/*create descriptor*/
	status = DftiCreateDescriptor(&hand, DFTI_SINGLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
	//if (0 != status) goto failed;

	/* Scaling */
	status = DftiSetValue(hand, DFTI_BACKWARD_SCALE,Scale);
	//if (0 != status) goto failed;

	/*Commiting descriptor*/
	status = DftiCommitDescriptor(hand);
	//if (0 != status) goto failed;

	/*Allocate temporary memory for one single subset in ifft*/
	temp = (MKL_Complex8*)mkl_malloc(N*sizeof(MKL_Complex8), 64);

	/*Initialize input for backward transform*/ 
	for (int i = 0; i < DATA_SIZE / N; i++)
	{
		/*Dividing the input into subsets in temp*/
		for (int j = 0; j < N; j++)
			temp[j] = x[j + i * N];
		/*Compute backward transform*/
		status = DftiComputeBackward(hand, temp);
		//if (0 != status) goto failed;
		/*The divided part in the Output Back to X*/
		for (int j = 0; j < N; j++)
			x[j + i * N] = temp[j];
	}
	/*
	//Verify the result of forward FFT if all are step functions
	
	for (int i = 0; i < DATA_SIZE; i++)
	{
		if (i % N == 0)
			printf("REAL%d = %f , IMAG%d = %f\n", i + 1, x[i].real, i + 1, x[i].imag);
	}
	*/

	/*Release the DFTI descriptor*/
	DftiFreeDescriptor(&hand);
	return status;
//failed:
//	status = 1;
//	goto END;
}