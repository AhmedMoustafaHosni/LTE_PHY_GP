/*This demo for executing DFT operation with the following configurations */
/*
Precision		: Single Precision 
Forward_Domain  : DFTI_COMPLEX
Dimension		: 1D
Placement		: INPLACE (means that the output array will replace the input)
Forward_scale   : 1 (The default)
*/

#include "Intel_Header.h"

int APP1 (void)
{
	/*Welcome message*/
	printf("APP1 is running \n");

	
	/* Size of 1D transform */
	int N = 14400;

	/* Pointer to input/output data */
	MKL_Complex8 *x = 0;

	/* Execution status */
	MKL_LONG status = 0;

	DFTI_DESCRIPTOR_HANDLE hand = 0;

	/*Create descriptor*/
	status = DftiCreateDescriptor(&hand, DFTI_SINGLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
	if (0 != status) goto failed;
	/*Commiting descriptor*/
	status = DftiCommitDescriptor(hand);
	if (0 != status) goto failed;

	/*Allocate input array*/
	x = (MKL_Complex8*)mkl_malloc(N*sizeof(MKL_Complex8), 64);
	if (0 == x) goto failed;

	/*Initialize input for forward transform*/
	
	for (int i = 0; i < N; i++) // Step input 
	{
		x[i].real = 1;
		x[i].imag = 0;
	}
	

	/*Compute forward transform*/
	double s_initial = 0, s_elapsed = 0;
	s_initial = dsecnd();  // sample time
	status = DftiComputeForward(hand, x);
	//if (0 != status) goto failed;
	s_elapsed = (dsecnd() - s_initial);
	printf(" completed with == \n == at %.5f milliseconds == \n\n", (s_elapsed * 1000));

	/*Verify the result of forward FFT*/
	
	for (int i = 0; i < N; i++)
	{
	//	printf("REAL%d = %f , IMAG%d = %f\n",i+1,x[i].real,i+1,x[i].imag);
	}
	

cleanup:

	/*Release the DFTI descriptor*/
	DftiFreeDescriptor(&hand);

	/*Free data array*/
	mkl_free(x);
	return status;

failed:
	status = 1;
	goto cleanup;
}
