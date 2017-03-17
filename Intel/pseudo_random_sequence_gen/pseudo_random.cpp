#include "Header.h"

unsigned short* pseudo_random_sequence_gen(int c_init, int seq_length)
{
	unsigned short * x1 = (unsigned short*)calloc((Nc + seq_length),sizeof(short));
	unsigned short * x2 = (unsigned short*)calloc((Nc + seq_length),sizeof(short));
	// initialization of x1 & x2 
	x1[0] = 1;
	int temp = c_init; 
	int i = 0;
	while (temp != 0)
	{
		x2[i] = temp % 2;
		temp /= 2;
		i++;
	}
	// generate the remaining part of x1 and x2 sequences
	int cond = seq_length + Nc - 31;
	for (int j = 0; j < cond; j++)
	{
		x1[j + 31] = (x1[j] + x1[j + 3]) % 2;
		x2[j + 31] = (x2[j] + x2[j+1] + x2[j+2] + x2[j+3]) % 2;
	}

	// generate c 
	unsigned short * C = (unsigned short*)calloc(seq_length,sizeof(short));
	for (int j = 0; j < seq_length; j++)
	{
		C[j] = (x1[j + Nc] + x2[j + Nc]) % 2;
	}
	return C;
}