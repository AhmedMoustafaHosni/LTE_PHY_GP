/*******************************************************************************
* Function:    SC_FDMA_demod
* Description: Single Carrier FDMA demodulator
* Inputs:     pusch_bb          - Modulated time domain signal
*
* Outputs:    N_ul_rb			- Number of RBs assigned to the ue
*			  input_subframe    - Gride 
*
*
* by: Khaled Ahmed Ali 
********************************************************************************/

#include "Intel_siso.h"


void SC_FDMA_demod(_MKL_Complex8* pusch_bb, int N_ul_rb, _MKL_Complex8 **  input_subframe)
{
	/* FFT configurations */
	MKL_LONG status = 0;
	DFTI_DESCRIPTOR_HANDLE handler = 0; 
	int length = 2048;
	/*create descriptor*/
	status = DftiCreateDescriptor(&handler, DFTI_SINGLE, DFTI_COMPLEX, 1, (MKL_LONG)length);
	/*Commiting descriptor*/
	status = DftiCommitDescriptor(handler);
	/*Prepare the needed vecors for ifft [0 0 0 . . . values . . .  . 0 0 0 ]*/
	_MKL_Complex8 input_subframe_modified[14][2048];
	input_subframe_modified[0:14][0:2048].real = 0;
	input_subframe_modified[0:14][0:2048].imag = 0;
	int ord_len = length + N_cp_L_else;
	int non_ord_len = length + N_cp_L_0;
	/* FFT input preparation and FFT execution */
	for (int i = 0; i < 14; i++)
	{
		/* Extraction of input */
		 switch(i)
		{
		case 1:case 2:case 3:case 4:case 5:case 6:
			//extraction 2048
			input_subframe_modified[i][:] = pusch_bb[non_ord_len + (i - 1) * ord_len + N_cp_L_else : length ];
			break;
		case 0:
			//extraction 2048
			input_subframe_modified[i][:] = pusch_bb[N_cp_L_0:length];
			break;
		case 7:
			//extraction 2048
			input_subframe_modified[i][:] = pusch_bb[non_ord_len + (i - 1) * ord_len + N_cp_L_0: length ];
			break;
		default:
			input_subframe_modified[i][:] = pusch_bb[2*non_ord_len + (i - 2) * ord_len + N_cp_L_else:length ];
		}
		/* Forward FFT */
		status = DftiComputeForward(handler, input_subframe_modified[i]);
	}

	/* FFT Shift to reorder the data */
	_MKL_Complex8 temp[14][2048 / 2];
	temp[0:14][0:length / 2] = input_subframe_modified[0:14][0:length / 2]; // store the first half in temp 
	input_subframe_modified[0:14][0:length / 2] = input_subframe_modified[0:14][length / 2:length / 2]; // replace the first haf with second half
	input_subframe_modified[0:14][length / 2:length / 2] = temp[0:14][0:length / 2]; //restore the first half in the place of the second half

	/* Extract the Data in SC  */
	length = N_ul_rb * N_sc_rb; // 1200
	int start = (FFT_size / 2) - (length / 2);
	input_subframe[0:14][0:length] = input_subframe_modified[0:14][start:length];
	/*verify*/
	/*
	FILE *fp;
	fp = fopen("Output.txt", "w");
	fprintf(fp, "REAL=%f  IMAG=%f\n", input_subframe[0:14][0:1200].real, input_subframe[0:14][0:1200].imag);
	*/
}