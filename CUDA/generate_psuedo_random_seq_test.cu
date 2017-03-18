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

int main(int argc, char **argv) {

	//For timing purpose
	float elapsed = 0;				//For time calc.
	cudaEvent_t start, stop;

	const int seq_len = 86400;
	const int n_RNTI = 10;
	const int n_s = 0;
	const int N_id_cell = 2;

	//startTimer();

	//For output:
	Byte *c_h = 0;

	//Call the Generation Function
	generate_psuedo_random_seq(&c_h, seq_len, n_RNTI, n_s, N_id_cell);

	//stopTimer("Time= %.10f ms\n", elapsed);

	//Print results
	//for (int i = 0; i < seq_len; i++)
	//	printf("idx = %d \t %d\n", i + 1, c_h[i]);

	//printf("\n\n");

	//To compare with MATLAB results
	//Run the file (seq_gen_Results.m)
	FILE *results;
	if ((results = freopen("seq_gen_Results.m", "w+", stdout)) == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}

	printf("clear; clc;\nc = [ ");
	for (int i = 0; i < seq_len; i++)
	{
		printf("%d", c_h[i]);
		if (i != (seq_len - 1))
			printf(",");
	}

	printf(" ];\n");

	//Matlab code
	printf("rng(10);\nc_init = 10 * 2 ^ 14 + floor(0 / 2) * 2 ^ 9 + 2;\nc_MATLAB = generate_psuedo_random_seq(c_init, %d);\nsum(abs(c-c_MATLAB))", seq_len);
	fclose(results);

}