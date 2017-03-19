/*******************************************************************************
* Function:    Interleaver
* Description: Interleaves ULSCH data with RI control information
* Inputs:      bits          - Binary digits to map
*	       bits_length   - length of input bits
*	       ri_bits       - RI control bits to interleave
*	       ribits_length - length of RI control bits
*	       mod           - modulation type (2 = QPSK, 4 = 16-QAM, 6= 64-QAM)
*
* Outputs:     out        - Interleaved bits stream of length (R_mux * C_mux)
*	       out_length - Length of the output stream
*
* Average Timing: 0.008 msecs
* Max timing (for 86400 bits and 64QAM): 0.13 msecs
*
* by: Mohammed Osama
********************************************************************************/

#include "Intel_siso.h"

void interleaver(float* bits, int bits_length, float* ri_bits, int ribits_length, float*& out, int& out_length, char mod)
{
	int H_prime = bits_length / mod;
	int N_ri_bits = ribits_length / mod;
	int H_prime_total = H_prime + N_ri_bits;
	int R_prime_mux = H_prime_total / C_mux;
	int R_mux = R_prime_mux * mod;


	out_length = C_mux*R_mux;

	// initialize matrix to all zeros
	float* mat = (float*)calloc(out_length, sizeof(float));
	char* y_idx = (char*)calloc(C_mux*R_prime_mux, sizeof(char));

	// allocating memory for the interleaved bits

	out = (float*)malloc(out_length * sizeof(float));

	// Adding RI bits
	char RI_column_set[4] = { 1,4,7,10 };
	int m = 0;
	int r;
	char C_ri;
	for (int n = 0; n < N_ri_bits; n++) {
		r = R_prime_mux - 1 - floor(n / 4);
		C_ri = RI_column_set[m];
		y_idx[r*C_mux + C_ri] = 1;
		for (int k = 0; k <= mod - 1; k++) {
			// ri_bits [ 2*row + col ], where 2 is the total number of columns which depend on mod order
			mat[C_mux*r*mod + C_ri*mod + k] = ri_bits[mod*n + k]; // converting ri_bits matrix into 1D stream
		}
		m = (m + 3) % 4;
	}

	// interleave the data
	int n = 0, k = 0;
	while (k < H_prime) {
		if (y_idx[n] == 0) {
			y_idx[n] = 1;
			for (int m = 0; m <= mod - 1; m++) {
				mat[n*mod + m] = bits[mod*k + m];
			}
			k = k + 1;
		}
		n = n + 1;
	}

	// Adding data

	int idx = 0;

	for (int i = 0; i <= C_mux - 1; i++)
		for (int j = 0; j <= R_prime_mux - 1; j++)
			for (int k = 0; k <= mod - 1; k++) {
				out[idx] = mat[i*mod + j*C_mux*mod + k];
				idx++;
			}

}
