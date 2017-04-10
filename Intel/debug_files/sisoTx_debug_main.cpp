/*
*  LTE siso chain
*  Max timing: 1.12 ms
*
*  Merge by Mohammed Osama
*/

#include "Intel_siso.h"


int main()
{
	unsigned char subframe_number = 0;		  // frame number
	unsigned char n_s = subframe_number * 2;  // slot number
	unsigned int cell_ID = 2;
	unsigned char delta_ss = 0;
	unsigned char cyclic_shift = 0;
	unsigned char cyclic_shift_dci = 0;
	unsigned char RBs_number = RESOURCE_BLOCKS;
	
	// initialize bits 
	float* bits = (float*) calloc(INTRLV, sizeof(float));  // input bits
	float* ri_bits = (float* ) malloc(N_RI_bits * MOD * sizeof(float));	
	float* out = NULL;   // output of interleaver
	int out_length;

	float* c = NULL;  //pseudo-random sequence
	
	__m256 result[FRAME_LENGTH / DataTypeLength]; // Result variable to store result of scrambler

	MKL_Complex8* symbols = (MKL_Complex8*) malloc(FRAME_LENGTH / MOD * sizeof(MKL_Complex8*));  // output symbols from mapper

	MKL_Complex8* dmrs1 = NULL;
	MKL_Complex8* dmrs2 = NULL;

	int SC = M_PUSCH_SC;

	MKL_Complex8 ** gride_out = NULL;

	MKL_Complex8* pusch_bb =(MKL_Complex8*) malloc(30720 * sizeof(MKL_Complex8*));

	// intialize Data and RI_bits
	for (int i = 0; i <= INTRLV - 1; i += 4)
	{
		bits[i] = 1;
		bits[i + 1] = 1;
		bits[i + 2] = 0;
		bits[i + 3] = 1;
	}
	for (int i = 0; i <= N_RI_bits * MOD - 1; i += 4)
	{
		ri_bits[i] = 0;
		ri_bits[i + 1] = 1;
		ri_bits[i + 2] = 0;
		ri_bits[i + 3] = 0;

	}

	FILE * fp;
	fp = fopen("output.m", "w");
	/* Interleaver */
	fprintf(fp, "bits = [");
	fprintf(fp, "%f ", bits[0:INTRLV]);
	fprintf(fp, "];\n");
	
	interleaver(bits, INTRLV, ri_bits, N_RI_bits*MOD, out, out_length, MOD);
	fprintf(fp, "b_intrlv = [");
	fprintf(fp, "%f ", out[0:out_length]);
	fprintf(fp, "];\n");
	
	int c_init = n_RNTI * 16384 + floor((float)n_s / 2) * 512 + cell_ID;
	c = pseudo_random_sequence_gen(c_init, out_length);

	/* Scrambler */
	for (int i = 0; i < out_length / DataTypeLength; i++)
	{
		__m256 vect_in1 = _mm256_setr_ps(out[LEN * i], out[LEN * i + 1], out[LEN * i + 2], out[LEN * i + 3], out[LEN * i + 4], out[LEN * i + 5], out[LEN * i + 6], out[LEN * i + 7]);// , out[LEN * i + 8], out[LEN * i + 9], out[LEN * i + 10], out[LEN * i + 11], out[LEN * i + 12], out[LEN * i + 13], out[LEN * i + 14], out[LEN * i + 15], out[LEN * i + 16], out[LEN * i + 17], out[LEN * i + 18], out[LEN * i + 19], out[LEN * i + 20], out[LEN * i + 21], out[LEN * i + 22], out[LEN * i + 23], out[LEN * i + 24], out[LEN * i + 25], out[LEN * i + 26], out[LEN * i + 27], out[LEN * i + 28], out[LEN * i + 29], out[LEN * i + 30], out[LEN * i + 31]);
		__m256 vect_in2 = _mm256_setr_ps(c[LEN * i], c[LEN * i + 1], c[LEN * i + 2], c[LEN * i + 3], c[LEN * i + 4], c[LEN * i + 5], c[LEN * i + 6], c[LEN * i + 7]);// , c[LEN * i + 8], c[LEN * i + 9], c[LEN * i + 10], c[LEN * i + 11], c[LEN * i + 12], c[LEN * i + 13], c[LEN * i + 14], c[LEN * i + 15], c[LEN * i + 16], c[LEN * i + 17], c[LEN * i + 18], c[LEN * i + 19], c[LEN * i + 20], c[LEN * i + 21], c[LEN * i + 22], c[LEN * i + 23], c[LEN * i + 24], c[LEN * i + 25], c[LEN * i + 26], c[LEN * i + 27], c[LEN * i + 28], c[LEN * i + 29], c[LEN * i + 30], c[LEN * i + 31]);
		result[i] = _mm256_xor_ps(vect_in1, vect_in2);
	}

	float* ptr = (float *)& result[0];
	
	fprintf(fp, "b_scrambled = [");
	fprintf(fp, "%f ", ptr[0:out_length]);
	fprintf(fp, "];\n");
	
	/* mapper */
	mapper(ptr, out_length, symbols, MOD);
	
	fprintf(fp, "mapped = [");
	fprintf(fp, "%f+1i*%f ", symbols[0:out_length / MOD].real, symbols[0:out_length / MOD].imag);
	fprintf(fp, "];\n");
	
	/** Generate dmrs **/
	generate_dmrs(subframe_number, cell_ID, delta_ss, cyclic_shift, cyclic_shift_dci, RBs_number, dmrs1, dmrs2);
	fprintf(fp, "dmrs1 = [");
	fprintf(fp, "%f+1i*%f ", dmrs1[0:M_PUSCH_SC].real, dmrs1[0:M_PUSCH_SC].imag);
	fprintf(fp, "];\n");
	fprintf(fp, "dmrs2 = [");
	fprintf(fp, "%f+1i*%f ", dmrs2[0:M_PUSCH_SC].real, dmrs2[0:M_PUSCH_SC].imag);
	fprintf(fp, "];\n");
	
	/** transform precoder **/
	Transform_precoder(symbols, SC);
	fprintf(fp, "precoder = [");
	fprintf(fp, "%f+1i*%f ", symbols[0:out_length / MOD].real, symbols[0:out_length / MOD].imag);
	fprintf(fp, "];\n");
	
	/** compose subframe **/
	gride_out = compose_subframe(symbols, dmrs1, dmrs2, RBs_number);

	/** SCFDMA modulator **/
	SC_FDMA_mod(pusch_bb, RBs_number, gride_out);
	fprintf(fp, "pusch_bb = [");
	fprintf(fp, "%f+1i*%f ", pusch_bb[0:30720].real, pusch_bb[0:30720].imag);
	fprintf(fp, "];");
	
	return 0;

}
