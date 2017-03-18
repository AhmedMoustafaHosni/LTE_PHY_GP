
#include "dmrs.h"

int main()
{

	
	unsigned char subframe_number = 0;
	unsigned int cell_ID = 2;
	unsigned char delta_ss = 0;
	unsigned char cyclic_shift = 0;
	unsigned char cyclic_shift_dci = 0;
	unsigned char RBs_number = 100;
	MKL_Complex8* dmrs1 = NULL;
	MKL_Complex8* dmrs2 = NULL;

	double s_initial = 0, s_elapsed = 0;

	// measure timing 
	s_initial = dsecnd();

	generate_dmrs(subframe_number, cell_ID, delta_ss, cyclic_shift, cyclic_shift_dci, RBs_number,dmrs1, dmrs2);
	
	s_elapsed = (dsecnd() - s_initial);

	
	/*FILE* fp = NULL;
	fp = fopen("CdmrsGen.m", "w");
	fprintf(fp, "dmrs_1_c = [ ");
	fprintf(fp, "%lf + 1i * %lf ", dmrs1[0:1200].real, dmrs1[0:1200].imag);
	fprintf(fp, "];\n");
	fprintf(fp, "dmrs_2_c = [ ");
	fprintf(fp, "%lf + 1i * %lf ", dmrs2[0:1200].real, dmrs2[0:1200].imag);
	fprintf(fp, "];\n");
*/
	printf(" DMRS generation completed processing for 100 Resource Blocks (max number of reference symbols) at == \n"
		" == %.5f milliseconds == \n\n", (s_elapsed * 1000));
	
	return 0;
}