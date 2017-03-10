
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
	/*
	for (int m = 0; m < RBs_number * 12; m++)
	{
		cout << dmrs1[m].real <<"+j*"<< dmrs1[m].imag << endl;
	}

	*/
	printf(" processing completed on 1728 bits at == \n"
		" == %.5f milliseconds == \n\n", (s_elapsed * 1000));
	
	return 0;
}