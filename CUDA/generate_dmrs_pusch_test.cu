/*
% Function:    generate_dmrs_pusch
% Description: Generates LTE demodulation reference signal for PUSCH
% Inputs:      N_subfr                  - Subframe number within a radio frame
%              N_id_cell                - Physical layer cell identity
%              delta_ss                 - Configurable portion of the sequence-shift pattern for PUSCH (sib2 groupAssignmentPUSCH)
%              group_hopping_enabled    - Boolean value determining if group hopping is enabled (sib2 groupHoppingEnabled)
%              sequence_hopping_enabled - Boolean value determining if sequence hopping is enabled (sib2 sequenceHoppingEnabled)
%              cyclic_shift             - Broadcast cyclic shift to apply to base reference signal (sib2 cyclicShift)
%              cyclic_shift_dci         - Scheduled cyclic shift to apply to base reference signal
%              w_config                 - fixed or table
%              N_prbs                   - Number of PRBs used for the uplink grant
%              layer                    - Which diversity layer to generate reference signals for
% Outputs:     *dmrs1_h					- Demodulation reference signal for PUSCH
*dmrs2_h					- Demodulation reference signal for PUSCH
By: Mohammed Mostafa
*/


#include "generate_dmrs_pusch.cuh"


int main(int argc, char **argv) {

	//For output
	cufftComplex *dmrs1_h;
	cufftComplex *dmrs2_h;

	//Call the generate_dmrs_pusch Function
	generate_dmrs_pusch(0, 2, 0, 0, 0, 0, 0, "fixed", 6, 0, &dmrs1_h, &dmrs2_h);

	//Print results
	for (int i = 0; i < 72 ; i++)
	{
			printf("idx = %d \t %f \t %f \n", i + 1, dmrs1_h[i].x, dmrs1_h[i].y);
	}
	
	for (int i = 0; i < 72; i++)
	{
		printf("idx = %d \t %f \t %f \n", i + 1, dmrs2_h[i].x, dmrs2_h[i].y);
	}

	//To compare with MATLAB results
	//Run the file (Demapper_Results.m)
	FILE *results;
	if ((results = freopen("dmrs_Results.m", "w+", stdout)) == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}

	//output file
	printf("clear; clc;\ndmrs1_real = [ ");
	for (int i = 0; i < (72); i++)
	{
		printf("%10f", dmrs1_h[i].x);
		if (i != ((72) - 1))
			printf(",");
	}

	printf(" ];\ndmrs1_imag = [ ");

	for (int i = 0; i < (72); i++)
	{
		printf("%10f", dmrs1_h[i].y);
		if (i != ((72) - 1))
			printf(",");
	}

	printf(" ];\n");
	printf("dmrs1_CUDA = dmrs1_real + 1i * dmrs1_imag;\n");


	printf("\ndmrs2_real = [ ");
	for (int i = 0; i < (72); i++)
	{
		printf("%10f", dmrs2_h[i].x);
		if (i != ((72) - 1))
			printf(",");
	}

	printf(" ];\ndmrs2_imag = [ ");

	for (int i = 0; i < (72); i++)
	{
		printf("%10f", dmrs2_h[i].y);
		if (i != ((72) - 1))
			printf(",");
	}

	printf(" ];\n");
	printf("dmrs2_CUDA = dmrs2_real + 1i * dmrs2_imag;\n");


	fclose(results);

	

	return 0;

}