/* Test both SC-FDMA MOD & DEM together*/
#include "Intel_Header.h"

int main()
{
	/*inputs*/
	int M_pusch_rb = 100;
	_MKL_Complex8 input_subframe[14][1200]; // 14 symboles in one subfram - 1200 subcarrier assigned for this single user
	//just for test
	input_subframe[:][0:1200].real = 1;
	input_subframe[:][:].imag = 0;
	_MKL_Complex8 pusch_bb[30720]; // the output

	/*modulator*/
	SC_FDMA_mod(pusch_bb, M_pusch_rb, input_subframe);

	double s_initial = 0, s_elapsed = 0;
	s_initial = dsecnd();  // sample time
	/*demodulator*/
	SC_FDMA_demod(pusch_bb, M_pusch_rb, input_subframe);
	s_elapsed = (dsecnd() - s_initial);
	printf(" completed with == \n == at %.5f milliseconds == \n\n", (s_elapsed * 1000));
	return 0;
}