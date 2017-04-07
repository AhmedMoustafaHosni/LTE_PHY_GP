
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
		// mapping of bits to variable names: 000000 ---> x1 x2 x3 x4 x5 x6
		// real: x1 x3 x5
		// imag: x2 x4 x6
		// general eqn: (-2*x1 + 1) * (a1*x2 + a2*x3 + a3*x2*x3 + b1) 
		for (int i = 0; i < bits_length / 6; i++) {
			// real = (-2 * x1 +1) * (6/sqrt(42) * x3 + 2/sqrt(42) * x5 - 1/sqrt(42))
			symbols[i].real = (-2 * bits[j] + 1) * (bits[j + 2] * NORM_64_2 + bits[j + 4] * -NORM_64_2 + bits[j + 2] * bits[j + 4] * NORM_64_4 + NORM_64_3);
			// imag = (-2 * x2 +1) * (6/sqrt(42) * x4 + 2/sqrt(42) * x6 - 1/sqrt(42))
			symbols[i].imag = (-2 * bits[j+1] + 1) * (bits[j + 3] * NORM_64_2 + bits[j + 5] * -NORM_64_2 + bits[j + 3] * bits[j + 5] * NORM_64_4 + NORM_64_3);
			j = j + 6;
		}
		break;
	default:
		break;
	}
	
}