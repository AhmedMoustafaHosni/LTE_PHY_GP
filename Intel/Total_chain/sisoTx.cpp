/*******************************************************************************
* Function:    LTE siso TX chain
* Description: Generates LTE SCFDMA signal for PUSCH
*
* Inputs:  bits               - Binary digits to map
*	       bits_length        - length of input bits
*	       ri_bits            - RI control bits
*	       ribits_length      - length of RI control bits
*	       mod                - modulation type (2 = QPSK, 4 = 16-QAM, 6= 64-QAM)
*		   N_rb               - Number of RBs assigned for the UE
*          N_subframe         - Subframe number within a radio frame
*		   cell_id            - Physical layer cell identity
*          delta_ss           - Configurable portion by higher layer of the sequence-shift pattern for PUSCH --> {0,1,...,29}
*		   cyclic_shift       - cyclic shift to apply to base reference signal --> {0,1,2,...,7}
*		   cyclic_shift_dci   - Scheduled cyclic shift to apply to base reference signal --> {0,1,2,...,7}
*
*
* Outputs: pusch_bb           - baseband SCFDMA symbols 
*
* by: Mohammed Osama
********************************************************************************/

#include "Intel_siso.h"

void sisoTx(_MKL_Complex8* pusch_bb, float* bits, int bits_length, float* ri_bits, int ribits_length, char mod, unsigned char N_rb, unsigned char N_suframe, unsigned int cell_id, unsigned char delta_ss, unsigned char cyclic_shift, unsigned char cyclic_shift_dci)
{

	/* Interleaver */
	float* out = NULL;   // output of interleaver
	int out_length;
	interleaver(bits, bits_length, ri_bits, ribits_length, out, out_length, mod);

	/* Pseudo-random sequence generation */
	unsigned char n_s = N_suframe * 2;
	int c_init = n_RNTI * 16384 + floor((float)n_s / 2) * 512 + cell_id;
	float* c = pseudo_random_sequence_gen(c_init, out_length);


	/* Scrambler */
	__m256 result[FRAME_LENGTH / DataTypeLength];  // store result of scrambler
	for (int i = 0; i < out_length / DataTypeLength; i++)
	{
		__m256 vect_in1 = _mm256_setr_ps(out[LEN * i], out[LEN * i + 1], out[LEN * i + 2], out[LEN * i + 3], out[LEN * i + 4], out[LEN * i + 5], out[LEN * i + 6], out[LEN * i + 7]);// , out[LEN * i + 8], out[LEN * i + 9], out[LEN * i + 10], out[LEN * i + 11], out[LEN * i + 12], out[LEN * i + 13], out[LEN * i + 14], out[LEN * i + 15], out[LEN * i + 16], out[LEN * i + 17], out[LEN * i + 18], out[LEN * i + 19], out[LEN * i + 20], out[LEN * i + 21], out[LEN * i + 22], out[LEN * i + 23], out[LEN * i + 24], out[LEN * i + 25], out[LEN * i + 26], out[LEN * i + 27], out[LEN * i + 28], out[LEN * i + 29], out[LEN * i + 30], out[LEN * i + 31]);
		__m256 vect_in2 = _mm256_setr_ps(c[LEN * i], c[LEN * i + 1], c[LEN * i + 2], c[LEN * i + 3], c[LEN * i + 4], c[LEN * i + 5], c[LEN * i + 6], c[LEN * i + 7]);// , c[LEN * i + 8], c[LEN * i + 9], c[LEN * i + 10], c[LEN * i + 11], c[LEN * i + 12], c[LEN * i + 13], c[LEN * i + 14], c[LEN * i + 15], c[LEN * i + 16], c[LEN * i + 17], c[LEN * i + 18], c[LEN * i + 19], c[LEN * i + 20], c[LEN * i + 21], c[LEN * i + 22], c[LEN * i + 23], c[LEN * i + 24], c[LEN * i + 25], c[LEN * i + 26], c[LEN * i + 27], c[LEN * i + 28], c[LEN * i + 29], c[LEN * i + 30], c[LEN * i + 31]);
		result[i] = _mm256_xor_ps(vect_in1, vect_in2);
	}
	float* ptr = (float *)& result[0];

	/* mapper */
	MKL_Complex8* symbols = (MKL_Complex8*)malloc(out_length / mod * sizeof(MKL_Complex8*));  // output symbols from mapper
	mapper(ptr, out_length, symbols, mod);

	/** Generate dmrs **/
	MKL_Complex8* dmrs1 = NULL;
	MKL_Complex8* dmrs2 = NULL;
	generate_dmrs(N_suframe, cell_id, delta_ss, cyclic_shift, cyclic_shift_dci, N_rb, dmrs1, dmrs2);

	/** transform precoder **/
	Transform_precoder(symbols, N_rb * 12);

	/** compose subframe **/
	MKL_Complex8 ** gride_out = NULL;
	gride_out = compose_subframe(symbols, dmrs1, dmrs2, N_rb);

	/** SCFDMA modulator **/
	SC_FDMA_mod(pusch_bb, N_rb, gride_out);

}
