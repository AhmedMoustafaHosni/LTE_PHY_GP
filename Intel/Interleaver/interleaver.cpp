/*******************************************************************************
* Function:    Interleaver
* Description: Interleaves ULSCH data with RI control information
* Inputs:      bits     - Binary digits to map
*			   ri_bits  - RI control bits to interleave
* 
* Outputs:     out      - Interleaved bits stream of size 1 * (R_mux * C_mux)  
*
*
* Average Timing: 0.026 msecs
* Max timing (for 86400 bits and 64QAM): 0.1 msecs
*
* by: Mohammed Osama
********************************************************************************/

#include "Intel_Header.h"

void interleaver (char* bits, char* ri_bits, char* out)
{
	// initialize matrix to all zeros
	char mat[C_mux*R_mux] = { 0 };
	char y_idx[C_mux*R_prime_mux];

	for (int i = 0; i <= C_mux*R_prime_mux - 1; i++)
	{
		y_idx[i] = -1;
	}

	// Adding RI bits
	char RI_column_set[4] = { 1,4,7,10 };
	int m = 0;
	int r;
	char C_ri;
	for (int n = 0; n < N_RI_bits; n++) {
		r = R_prime_mux - 1 - floor(n / 4);
		C_ri = RI_column_set[m];
		y_idx[r*C_mux + C_ri] = 1;
		for (int k = 0; k <= Q_M - 1; k++) {
			// ri_bits [ 2*row + col ], where 2 is the total number of columns which depend on mod order
			mat[C_mux*r*Q_M + C_ri*Q_M + k] = ri_bits[Q_M*n + k]; // converting ri_bits matrix into 1D stream
		}
		m = (m + 3) % 4;
	}

	// interleave the data
	int n = 0, k = 0;
	while (k < H_prime) {
		if (y_idx[n] == -1) {
			y_idx[n] = 1;
			for (int m = 0; m <= Q_M - 1; m++) {
				mat[n*Q_M + m] = bits[Q_M*k + m];
			}
			k = k + 1;
		}
		n = n + 1;
	}

	// Adding data

	int idx = 0;

	for (int i = 0; i <= C_mux - 1; i++)
		for (int j = 0; j <= R_prime_mux - 1; j++)
			for (int k = 0; k <= Q_M - 1; k++) {
				out[idx] = mat[i*Q_M + j*C_mux*Q_M + k];
				idx++;
	}

}