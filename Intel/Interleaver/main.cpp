
/* testing interleaver using Data bits and RI bits   */
#include "Intel_Header.h"

int main()
{
	// input bits
	char bits[FRAME_LENGTH];

	// input RI bits
	char ri_bits[N_RI_bits * Q_M];

	// intialize Data and RI_bits
	for (int i = 0; i <= FRAME_LENGTH - 1; i += 4)
	{
		bits[i] = 1;
		bits[i + 1] = 1;
		bits[i + 2] = 0;
		bits[i + 3] = 1;

	}
	for (int i = 0; i <= N_RI_bits * Q_M - 1; i += 4)
	{
		ri_bits[i] = 0;
		ri_bits[i + 1] = 1;
		ri_bits[i + 2] = 0;
		ri_bits[i + 3] = 0;

	}
	double s_initial = 0, s_elapsed = 0;

	// measure timing 
	s_initial = dsecnd();
	/****************************************** Interleaver ********************************************/

	char out[C_mux*R_mux];
	interleaver(bits, ri_bits, out);
	
	/******************************************    END      ********************************************/
	s_elapsed = (dsecnd() - s_initial);

	/* printing output */
	cout << "start ";
	for (int i = 0; i <= FRAME_LENGTH/2 + 100; i++) {
	cout << (int) out[i] << endl;
	}

	for (int i = 965; i <= C_mux*R_mux - 1; i++) {
		cout << (int)out[i] << endl;
	}
	
	printf(" processing completed on 1728 bits at == \n"
		" == %.5f milliseconds == \n\n", (s_elapsed * 1000));

}