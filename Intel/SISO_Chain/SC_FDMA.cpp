#include "Intel_siso.h"

void SC_FDMA_mod(_MKL_Complex8* pusch_bb, int M_pusch_rb, _MKL_Complex8** input_subframe)
{

	/*sc-fdma parameters based into the input*/
	//int N_ul_rb = M_pusch_rb;
	int length = M_pusch_rb * N_sc_rb;
	int start = (FFT_size / 2) - (length / 2);
	int Num_symbols_subfram = 14;

	/*Prepare the needed vecors for ifft [0 0 0 . . . values . . .  . 0 0 0 ]*/
	_MKL_Complex8 input_subframe_modified[14][2048];
	input_subframe_modified[:][:].real = 0;
	input_subframe_modified[:][:].imag = 0;
	input_subframe_modified[:][start:length] = input_subframe[0:Num_symbols_subfram][0:length];
	length = 2048;
	/* perform FFTSHIFT which is the swap between the first half and the second half of each symbol*/
	_MKL_Complex8 temp[14][2048 / 2];
	temp[:][:] = input_subframe_modified[:][0:length / 2]; // store the first half in temp 
	input_subframe_modified[:][0:length / 2] = input_subframe_modified[:][length / 2:length / 2]; // replace the first haf with second half
	input_subframe_modified[:][length / 2:length / 2] = temp[:][:]; //restore the first half in the place of the second half
	/*Perform the FFT operation */
	
	/* Scaling IDFT*/
	float Scale = 1 / float(length);
	/* Execution status */
	MKL_LONG status = 0;
	DFTI_DESCRIPTOR_HANDLE hand = 0;
	/*create descriptor*/
	status = DftiCreateDescriptor(&hand, DFTI_SINGLE, DFTI_COMPLEX, 1, (MKL_LONG)length);
	/* Scaling */
	status = DftiSetValue(hand, DFTI_BACKWARD_SCALE, Scale);
	/*Commiting descriptor*/
	status = DftiCommitDescriptor(hand);
	int ord_len = length + N_cp_L_else;
	int non_ord_len = length + N_cp_L_0;
	/*Compute forward transform*/
	double s_initial_test = 0, s_elapsed_test = 0;

	for (int i = 0; i < 14; i++)
	{
		/* IFFT */
		status = DftiComputeBackward(hand, input_subframe_modified[i]);
		/* Compose the time domain Serial signal  */
		switch (i)
		{
		case 1:case 2:case 3:case 4:case 5:case 6:
			//CP
			pusch_bb[non_ord_len + (i - 1) * ord_len:N_cp_L_else] = input_subframe_modified[i][length - N_cp_L_else:N_cp_L_else];
			//DATA
			pusch_bb[non_ord_len + (i - 1) * ord_len + N_cp_L_else:length] = input_subframe_modified[i][0:length];
			break;
		case 0:
			//CP SYMB_0
			pusch_bb[0:N_cp_L_0] = input_subframe_modified[0][length - N_cp_L_0:N_cp_L_0];
			//DATA SYMB_0
			pusch_bb[N_cp_L_0:length] = input_subframe_modified[0][0:length];
			break;
		case 7:
			//CP SYMB_7
			pusch_bb[non_ord_len + (i - 1) * ord_len:N_cp_L_0] = input_subframe_modified[i][length - N_cp_L_0:N_cp_L_0];
			//DATA SYMB_7
			pusch_bb[non_ord_len + (i - 1) * ord_len + N_cp_L_0:length] = input_subframe_modified[i][0:length];
			break;
		default:
			//CP
			pusch_bb[2 * non_ord_len + (i - 2) * ord_len:N_cp_L_else] = input_subframe_modified[i][length - N_cp_L_else:N_cp_L_else];
			//DATA
			pusch_bb[2 * non_ord_len + (i - 2) * ord_len + N_cp_L_else:length] = input_subframe_modified[i][0:length];
			break;
		}
	}

}