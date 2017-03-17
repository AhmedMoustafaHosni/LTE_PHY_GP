#include "Header.h"

int main()
{

	int c_init = n_RNTI * 16384 + floor((float)n_s / 2) * 512 + N_id_cell;
	int seq_length = 86400;

	//time calculation & calling the function
	double s_initial = 0, s_elapsed = 0;
	s_initial = dsecnd();  // sample time	

	unsigned short* code = pseudo_random_sequence_gen(c_init, seq_length);

	s_elapsed = (dsecnd() - s_initial);
	printf(" completed task1 with == \n == at %.5f milliseconds == \n\n", (s_elapsed * 1000));


	// output to be tested with matlab
	FILE* fp = NULL;
	fp = fopen("outputcode.m", "w");
	fprintf(fp, "code_from_c = [ ");
	fprintf(fp, "%d ,", code[0:seq_length]);
	fprintf(fp, "];");
	return 0;
}