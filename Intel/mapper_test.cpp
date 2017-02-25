/*******************************************************************************
* Function:    modulation_mapper
* Description: Maps binary digits to complex-valued modulation symbols (support QPSK only till now)
* Inputs:      bits     - Binary digits to map
* Outputs:     symbols  - Complex-valued modulation symbols in MKL_Complex8 structure
* 
* Average Timing: 0.0073 msecs
********************************************************************************/

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <mkl.h>

#define FRAME_LENGTH 1728

using namespace std;

int main()
{
	// Intializing input array
	float bits[FRAME_LENGTH];
	// Intializing input array
	for (int i = 0; i < FRAME_LENGTH; i++)
	{
		bits[i] = 1;
	}
	
	// measure timing 
	double s_initial = 0, s_elapsed = 0;
	s_initial = dsecnd();

	/////////////////////////////////// Starting of QPSK code\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
	// mapping bits to symbols
	// bits*-1.4142 + 0.7071 
	MKL_Complex8 symbols [FRAME_LENGTH/2];

	int j = 0;
	for (int i = 0; i < FRAME_LENGTH / 2; i++) {
		symbols[i].real = bits[j] * -1.4142 + 0.7071;
		symbols[i].imag = bits[j + 1] * -1.4142 + 0.7071;
		j = j+2;
	}

	/////////////////////////////////////// END \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ 
	s_elapsed = (dsecnd() - s_initial);

	for (int i = 0; i < FRAME_LENGTH / 2; i++) {
		cout << symbols[i].real << " +j " << symbols[i].imag << endl;
		//printf("%.4f +j %.4f\n", symbols[i].real, symbols[i].imag);
	}

	printf(" completed == \n"
		" == at %.5f milliseconds == \n\n", (s_elapsed * 1000));
	return 0;
}
