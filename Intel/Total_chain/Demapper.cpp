/*-------------------------------
Function:    demapper

Description : Maps complex - valued modulation symbols to binary digits using hard decision

Inputs :
-symbols:	pointer to array of MKL complex symbols
-length:	lenghth of the symbols array
-mod_type:	Modulation type(bpsk = 1 , qpsk = 2, 16qam  = 4,or 64qam = 6)
bits:		array of demodulated bits

edit 1 / 3 / 2017
by Ahmed Moustafa
------------------------------------*/


#include "Intel_siso.h"


#define sqrt_10 0.632455532f
#define sqrt_42_2 0.3086066999f
#define sqrt_42_4 0.6172133998f
#define sqrt_42_6 0.9258200998f




void demapper(MKL_Complex8 *symbols, int length, int mod_type, float *bits)
{
	int i = 0;

	switch (mod_type)
	{

	case 1:
		for (; i < length; i++)
		{
			if (symbols[i].real > 0)
			{
				bits[i] = 0;
			}
			else
			{
				bits[i] = 1;
			}
		}
		break;


	case 2:
		for (; i < length; i++)
		{
			if (symbols[i].real > 0)
			{
				bits[2 * i] = 0;
			}
			else
			{
				bits[2 * i] = 1;
			}
			if (symbols[i].imag > 0)
			{
				bits[(2 * i) + 1] = 0;
			}
			else
			{
				bits[(2 * i) + 1] = 1;
			}
		}
		break;



	case 4:
		for (; i < length; i++)
		{

			if (symbols[i].real < 0)
			{
				bits[4 * i] = 1;
			}
			else
			{
				bits[4 * i] = 0;
			}


			if (symbols[i].imag < 0)
			{
				bits[(4 * i) + 1] = 1;
			}
			else
			{
				bits[(4 * i) + 1] = 0;
			}

			if (fabs(symbols[i].real) < sqrt_10)
			{
				bits[(4 * i) + 2] = 0;
			}
			else
			{
				bits[(4 * i) + 2] = 1;
			}

			if (fabs(symbols[i].imag) < sqrt_10)
			{
				bits[(4 * i) + 3] = 0;
			}
			else
			{
				bits[(4 * i) + 3] = 1;
			}
		}
		break;



	case 6:
		for (; i < length; i++)
		{

			if (symbols[i].real < 0)
			{
				bits[6 * i] = 1;
			}
			else
			{
				bits[6 * i] = 0;
			}


			if (symbols[i].imag < 0)
			{
				bits[(6 * i) + 1] = 1;
			}
			else
			{
				bits[(6 * i) + 1] = 0;
			}

			if (fabs(symbols[i].real) < sqrt_42_4)
			{
				bits[(6 * i) + 2] = 0;
			}
			else
			{
				bits[(6 * i) + 2] = 1;
			}

			if (fabs(symbols[i].imag) < sqrt_42_4)
			{
				bits[(6 * i) + 3] = 0;
			}
			else
			{
				bits[(6 * i) + 3] = 1;
			}

			if ((fabs(symbols[i].real) > sqrt_42_2) && (fabs(symbols[i].real) < sqrt_42_6))
			{
				bits[(6 * i) + 4] = 0;
			}
			else
			{
				bits[(6 * i) + 4] = 1;
			}

			if ((fabs(symbols[i].imag) > sqrt_42_2) && (fabs(symbols[i].imag) < sqrt_42_6))
			{
				bits[(6 * i) + 5] = 0;
			}
			else
			{
				bits[(6 * i) + 5] = 1;
			}
		}
		break;


	default:
		return;

	}


}




