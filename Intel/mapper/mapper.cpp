/*******************************************************************************
* Function:    modulation_mapper
* Description: Maps binary digits to complex-valued modulation symbols (support QPSK only till now)
* Inputs:      bits     - Binary digits to map
* Outputs:     symbols  - Complex-valued modulation symbols in MKL_Complex8 structure
* 
* Max Timing: 0.075 msecs
* By Mohammed Osama
********************************************************************************/
#include "mapper.h"

void mapper(float* bits, int bits_length, MKL_Complex8* symbols, char mod)
{
	
	int j = 0;
	switch (mod)
	{
	case 2:
		for (int i = 0; i < bits_length / 2; i++) {
			symbols[i].real = bits[j] * -sqrt2 + sqrt2_rec;
			symbols[i].imag = bits[j + 1] * -sqrt2 + sqrt2_rec;
			j = j + 2;
		}
		break;
	case 4:
		// 16 qam
		break;
	case 6:
		// 64 qam
		break;
	default:
		break;
	}
	
}