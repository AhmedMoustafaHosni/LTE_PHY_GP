#include "Intel_siso.h"

unsigned short* pseudo_random_sequence_gen(int c_init, int seq_length)
{
	unsigned short * x1 = (unsigned short*)calloc((Nc + seq_length),sizeof(short));
	unsigned short * x2 = (unsigned short*)calloc((Nc + seq_length),sizeof(short));
	// initialization of x1 & x2 
	x1[0] = 1;
	
	for (int i = 0; i < 31; i++)
	{
		x2[i] = (c_init >> i) & 1;
	}
	
	
	// generate the remaining part of x1 and x2 sequences
	for (int n = 0; n < Nc + seq_length - 31; n++)
	{
		x1[n + 31] = x1[n + 3] ^ x1[n];
		x2[n + 31] = x2[n + 3] ^ x2[n + 2] ^ x2[n + 1] ^ x2[n];
	}

	// generate c 
	unsigned short * C = (unsigned short*)calloc(seq_length,sizeof(short));
	for (int n = 0; n < seq_length; n++)
	{
		C[n] = x1[n + Nc] ^ x2[n + Nc];
	}
	return C;
}