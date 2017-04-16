/*******************************************************************************
* Function:    Deinterleaver
* Description: Deinterleaves ULSCH data with RI control information
* Inputs:      bits      - interleaved data and control bits
*			   ri_bits   - RI control bits to interleave
*              N_ri_bits - multiples of 12 (length of RI bits = N_ri_bits * MOD)
*			   mod       - modulation type (2 = QPSK, 4 = 16-QAM, 6= 64-QAM)
*
* Outputs:     out          - Deinterleaved data bits stream
*			   out_ri_bits  - Deinterleaved RI control bits stream
*
* Average Timing: 0.007 msecs
* Max timing (for 86400 bits and 64QAM): 0.11 msecs
*
* by: Mohammed Osama
********************************************************************************/

#include "Intel_siso.h"

void deinterleaver(char* bits, int bits_length, char mod, char N_ri_bits, char* out_ri_bits, char* out)
{
	char RI_column_set[4] = { 1,4,7,10 };
	int H_Prime_Total = bits_length / mod;
	int H_Prime = H_Prime_Total - N_ri_bits;
	int  R_Prime_Mux = H_Prime_Total / C_mux;
	int R_Mux = R_Prime_Mux * mod;

	
	
    // initialize matrix to all zeros
	char* mat = (char*) calloc(C_mux*R_Mux, sizeof(char));
	char* y_idx = (char*) calloc(C_mux*R_Prime_Mux, sizeof(char));
	
	// move from bottom to top
	
	// Reconstruct matrix from the output of interleaver

	int idx = 0;
	for (int i = 0; i <= C_mux - 1; i++)
		for (int j = 0; j <= R_Prime_Mux - 1; j++)
			for (int k = 0; k <= mod - 1; k++) {
				mat[i*mod + j*C_mux*mod + k] = bits[idx];
				idx++;
			}

	// Deinterleave the RI control bits
	
	if (N_ri_bits == 0)
		out_ri_bits = NULL;  // no RI_bits
	else {
		int m = 0;
		int r;
		char C_ri;
		for (int n = 0; n < N_ri_bits; n++) {
			r = R_Prime_Mux - 1 - floor(n / 4);
			C_ri = RI_column_set[m];
			y_idx[r*C_mux + C_ri] = 1;
			for (int k = 0; k <= mod - 1; k++) {
				// ri_bits [ 2*row + col ], where 2 is the total number of columns which depend on mod order
				out_ri_bits[mod*n + k] = mat[C_mux*r*mod + C_ri*mod + k]; // converting ri_bits matrix into 1D stream
			}
			m = (m + 3) % 4;
		}
	}
	// interleave the data
	int n = 0, k = 0;
	while (k < H_Prime) {
		if (y_idx[n] == 0) {
			y_idx[n] = 1;
			for (int m = 0; m <= mod - 1; m++) {
				out[mod*k + m]= mat[n*mod + m];
			}
			k = k + 1;
		}
		n = n + 1;
	}

	
}