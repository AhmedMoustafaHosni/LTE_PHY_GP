/*
This function is used to read the bits from a text file and return the number of bits as well.
If no argument is passed, it will try to read from the same directory from the file "input.txt".
Bits should be in one line with no spaces, for example: (100011110)
To do so in MATLAB, use this command:

dlmwrite('output.txt',Variable_to_Print,'delimiter','');

By: Ahmad Nour
*/

#include "input.h"

int main(int argc, char **argv)
{
	
	int N;		//Number of bits in the file
	BYTE* inputBits = readBits(argc, argv[1], &N);

	for (int i = 0; i < N; i++)
		printf("%c", inputBits[i]);

	printf("\n%d bits were read\n", N);

}
