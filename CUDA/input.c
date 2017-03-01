/*
This function is used to read the bits from a text file and return the number of bits as well.
If no argument is passed, it will try to read from the same directory from the file "input.txt".
Bits should be in one line with no spaces, for example: (100011110)
To do so in MATLAB, use this command:

dlmwrite('output.txt',Variable_to_Print,'delimiter','');

By: Ahmad Nour
*/

#include "input.h"

BYTE* readBits(int argc, char* argv, int *numBits)
{

	FILE *inputFile;
	size_t readCount, N = 0;
	char* path = "input.txt";

	if (argc >= 2)		//a path is given, use it instead
	{
		path = argv;
	}

	if ((inputFile = fopen(path, "r+")) == NULL) {
		printf("Cannot open file.\n");
		exit(1);
	}

	char* inputBuffer = (char*)malloc(sizeof(char)* BUFF);
	char* inputBits = (char*)malloc(sizeof(char)* BUFF);

	while ((readCount = fread(inputBuffer, sizeof(char), BUFF, inputFile)) > 0)
	{
		inputBits = (unsigned char*)realloc(inputBits, readCount);
		N += readCount;
		for (int i = 0; i < readCount; i++)
			inputBits[i] = inputBuffer[i];
	}

	free(inputBuffer);
	fclose(inputFile);

	*numBits = N;
	return inputBits;

}
