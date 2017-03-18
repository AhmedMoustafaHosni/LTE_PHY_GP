/*-------------------------------
Function:    demapper

Description : Maps complex - valued modulation symbols to binary digits using hard decision

Inputs : 
-symbols:	pointer to array of MKL complex symbols
-length:	lenghth of the symbols array
-mod_type:	Modulation type(bpsk = 2 , qpsk = 4, 16qam  = 16,or 64qam = 64)
bits:		array of demodulated bits

edit 1 / 3 / 2017
by Ahmed Moustafa
------------------------------------*/

#pragma once


void demapper(MKL_Complex8 *symbols, int length, int mod_type, float *bits);



