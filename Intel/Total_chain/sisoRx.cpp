/*******************************************************************************
* Function:    LTE siso RX chain
* Description: Reconstract the data bits and RI-bits
*
* Inputs:  bits               - Binary digits to map
*	       bits_length        - length of input bits
*	       ribits_length      - length of RI control bits
*		   N_ul_rb            - Number of RBs assigned for the UE
*          N_subframe         - Subframe number within a radio frame
*		   cell_id            - Physical layer cell identity
*
* Outputs: out                - Deinterleaved data bits stream
*		   out_ri_bits        - Deinterleaved RI control bits stream*
*
* Note: Test sections are for matlab testing 
* by: Khaled Ahmed Ali
********************************************************************************/

#include "Intel_siso.h"

void sisoRx(_MKL_Complex8* pusch_bb, int N_ul_rb, float * bits, unsigned char N_suframe, unsigned int cell_id, int ribits_length, char* out_ri_bits, char* out_bits)
{
	//uncomment this section for data error test
	/*FILE * fp;
	fp = fopen("SC_DEMOD.txt", "w");*/
	/******************************Step 1****************************************/
	// SC_FDMA Demodulator
	MKL_Complex8 ** gride = (MKL_Complex8 **)malloc(NUM_SYM_SUBFRAME * sizeof(MKL_Complex8 *));
	for (int i = 0; i < NUM_SYM_SUBFRAME; i++)
	{
		gride[i] = (MKL_Complex8 *)malloc(N_ul_rb*N_sc_rb * sizeof(MKL_Complex8));
	}
	SC_FDMA_demod(pusch_bb, N_ul_rb, gride);
	/******************************EO-Step 1*******************************************/
	// test demodulator output
	/*
	fprintf(fp,"gride_Rx =[ ");
	for (int i = 0; i < 14; i++)
	{
		fprintf(fp, "%f+i*%f ", gride[i][0:1200].real, gride[i][0:1200].imag);
		fprintf(fp, ";");
	}
	fprintf(fp, "]; \n");*/
	/******************************Step 2****************************************/
	// Decompose subfram
	int number_of_data_symbols_in_subfram = 12;
	MKL_Complex8 * data = (MKL_Complex8 *)malloc(N_ul_rb*N_sc_rb*number_of_data_symbols_in_subfram*sizeof(MKL_Complex8));
	MKL_Complex8* dmrs_1 = (MKL_Complex8 *)malloc(N_ul_rb*N_sc_rb*sizeof(MKL_Complex8));
	MKL_Complex8* dmrs_2 = (MKL_Complex8 *)malloc(N_ul_rb*N_sc_rb*sizeof(MKL_Complex8));
	decompose_subframe(gride, data, dmrs_1, dmrs_2, N_ul_rb);
	/******************************EO-Step 2*******************************************/
	//test Decompose subfram
	
    /*fprintf(fp, "transform_precoder_in =[ ");
	fprintf(fp, "%f+i*%f ", data[0:14400].real, data[0:14400].imag);
	fprintf(fp, "]; \n");
*/
	/******************************Step 3****************************************/
	//Transform predecoder
	Transform_predecoder(data, N_ul_rb*N_sc_rb);
	/******************************EO-Step 3*******************************************/
	//test transform predecoder
/*
	fprintf(fp, "demapper_in =[ ");
	fprintf(fp, "%f+i*%f ", data[0:14400].real, data[0:14400].imag);
	fprintf(fp, "]; \n");*/
	/******************************Step 4****************************************/
	//Demapper
	int length = N_ul_rb*N_sc_rb*number_of_data_symbols_in_subfram;
	demapper(data, length, MOD, bits);
	/******************************EO-Step 4*******************************************/
	//test Demapper

	/*fprintf(fp, "scrambled_bits =[ ");
	fprintf(fp, "%f ", bits[0:FRAME_LENGTH]);
	fprintf(fp, "]; \n");
*/
	/******************************Step 5****************************************/
	//Descrambler & pseudo random sequence generator 
	unsigned char n_s = N_suframe * 2;
	int c_init = n_RNTI * 16384 + floor((float)n_s / 2) * 512 + cell_id;
	float* c = pseudo_random_sequence_gen(c_init, FRAME_LENGTH);
	__m256 result[FRAME_LENGTH / DataTypeLength];  // store result of scrambler
	for (int i = 0; i < FRAME_LENGTH / DataTypeLength; i++)
	{
		__m256 vect_in1 = _mm256_setr_ps(bits[LEN * i], bits[LEN * i + 1], bits[LEN * i + 2], bits[LEN * i + 3], bits[LEN * i + 4], bits[LEN * i + 5], bits[LEN * i + 6], bits[LEN * i + 7]);// , out[LEN * i + 8], out[LEN * i + 9], out[LEN * i + 10], out[LEN * i + 11], out[LEN * i + 12], out[LEN * i + 13], out[LEN * i + 14], out[LEN * i + 15], out[LEN * i + 16], out[LEN * i + 17], out[LEN * i + 18], out[LEN * i + 19], out[LEN * i + 20], out[LEN * i + 21], out[LEN * i + 22], out[LEN * i + 23], out[LEN * i + 24], out[LEN * i + 25], out[LEN * i + 26], out[LEN * i + 27], out[LEN * i + 28], out[LEN * i + 29], out[LEN * i + 30], out[LEN * i + 31]);
		__m256 vect_in2 = _mm256_setr_ps(c[LEN * i], c[LEN * i + 1], c[LEN * i + 2], c[LEN * i + 3], c[LEN * i + 4], c[LEN * i + 5], c[LEN * i + 6], c[LEN * i + 7]);// , c[LEN * i + 8], c[LEN * i + 9], c[LEN * i + 10], c[LEN * i + 11], c[LEN * i + 12], c[LEN * i + 13], c[LEN * i + 14], c[LEN * i + 15], c[LEN * i + 16], c[LEN * i + 17], c[LEN * i + 18], c[LEN * i + 19], c[LEN * i + 20], c[LEN * i + 21], c[LEN * i + 22], c[LEN * i + 23], c[LEN * i + 24], c[LEN * i + 25], c[LEN * i + 26], c[LEN * i + 27], c[LEN * i + 28], c[LEN * i + 29], c[LEN * i + 30], c[LEN * i + 31]);
		result[i] = _mm256_xor_ps(vect_in1, vect_in2);
	}
	float* ptr = (float *)& result[0];
	/******************************EO-Step 5*******************************************/
	//test descrambler

	/*fprintf(fp, "interleaved_bits =[ ");
	fprintf(fp, "%f ", ptr[0:FRAME_LENGTH]);
	fprintf(fp, "]; \n");
*/
	/******************************Step 6****************************************/
	// Deinterleaver
	char * Semi_final = (char *)malloc(FRAME_LENGTH* sizeof(char));
	Semi_final[0:FRAME_LENGTH] = ptr[0:FRAME_LENGTH];
	out_bits = (char*)calloc(INTRLV, sizeof(char));  // output bits
	out_ri_bits = (char*)malloc(N_RI_bits * MOD * sizeof(char));
	deinterleaver(Semi_final, FRAME_LENGTH, MOD, N_RI_bits, out_ri_bits, out_bits);
	/******************************EO-Step 6*******************************************/
	//test deinterleaver

	/*fprintf(fp, "out_bits =[ ");
	fprintf(fp, "%d ", out_bits[0:INTRLV]);
	fprintf(fp, "]; \n");*/
	/********************************************************************************/
}