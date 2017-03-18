/*
% Function:    generate_ul_rs
% Description: Generates LTE base reference signal sequence for the UL
% Inputs:      N_s                      - Slot number within a radio frame
%              N_id_cell                - Physical layer cell identity
%              chan_type                - pusch or pucch
%              delta_ss                 - Configurable portion of the sequence-shift pattern for PUSCH (sib2 groupAssignmentPUSCH)
%              group_hopping_enabled    - Boolean value determining if group hopping is enabled (sib2 groupHoppingEnabled)
%              sequence_hopping_enabled - Boolean value determining if sequence hopping is enabled (sib2 sequenceHoppingEnabled)
%              alpha                    - Cyclic shift
%              N_prbs                   - Number of PRBs used for the uplink grant
% Outputs:     ref_signal				- Base reference signal
By: Mohammed Mostafa
*/

#include "generate_dmrs_pusch.cuh"
#include "generate_ul_rs.cuh"

int main(int argc, char **argv) {

	//For output
	cufftComplex *dmrs1_h;


	//Call the generate_ul_rs Function
	generate_ul_rs(0, 2, "pusch", 0, 0, 0, 2.617993877991494, 6, &dmrs1_h);

	//Print results
	for (int i = 0; i < 72 ; i++)
	{
			printf("idx = %d \t %f \t %f \n", i + 1, dmrs1_h[i].x, dmrs1_h[i].y);
	}
	

	//To compare with MATLAB results
	//Run the file (Demapper_Results.m)
	FILE *results;
	if ((results = freopen("ul_rs_Results.m", "w+", stdout)) == NULL) {
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
	fclose(results);
	
	return 0;

}