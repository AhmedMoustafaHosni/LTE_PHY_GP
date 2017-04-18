/*
% Function:		generate_psuedo_random_seq
% Description:	Generates the psuedo random sequence c
% Inputs:		bits_h:					Binary bits to scramble
%				seq_len:				Length of the output sequence
%				n_RNTI:					= 10 --> radio network temporary identifier given to the UE by enodeB
%				n_s:					= 0 --> assume UE send on time slot 0
%				N_id_cell:				= 2 --> assume enodeB scheduled cell 2 for the UE
% Outputs:		*c_h:					Psuedo random sequence
By: Ahmad Nour
*/

#include "generate_psuedo_random_seq.cuh"

void generate_psuedo_random_seq(Byte** c_h, const int seq_len, const int n_RNTI, const int n_s, const int N_id_cell)
{

	const long c_init = (long)n_RNTI * pow(2, 14) + (n_s / 2) * pow(2, 9) + N_id_cell;

	*c_h = (Byte *)malloc(sizeof(Byte)*seq_len);
	Byte *x1_h = (Byte *)malloc(sizeof(Byte)*(Nc + seq_len));
	Byte *x2_h = (Byte *)malloc(sizeof(Byte)*(Nc + seq_len));

	x1_h[0] = 1;

	for (int i = 1; i < (Nc + seq_len); i++)
	{
		x1_h[i] = 0;
		x2_h[i] = 0;
	}

	for (int i = 0; i < 31; i++)
	{
		x2_h[i] = (c_init >> i) & 1;
	}

	for (int n = 0; n < Nc + seq_len - 31; n++)
	{
		x1_h[n + 31] = x1_h[n + 3] ^ x1_h[n];
		x2_h[n + 31] = x2_h[n + 3] ^ x2_h[n + 2] ^ x2_h[n + 1] ^ x2_h[n];
	}

	for (int n = 0; n < seq_len; n++)
	{
		(*c_h)[n] = x1_h[n + Nc] ^ x2_h[n + Nc];
	}
}