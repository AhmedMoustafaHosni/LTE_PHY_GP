#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>

//serial time 0.00663

#define DataLength 1728 // Number of bits 
#define DataTypeLength 8 // We can divide int to 16 or 8 elements in 256b vector

int main()
{ 
	float b[DataLength]; // input bits float to fit with mapper
	float c[DataLength]; // input c random sequence 
	
	//initialization of bits 
	for (int i = 0; i < DataLength; i++)
	{
		b[i] = 1;
		c[i] = 0;
	}

	__m256 result[DataLength / DataTypeLength]; // Result variable to store result
	double s_initial = 0, s_elapsed = 0;
	s_initial = dsecnd();  // sample time
	for (int i = 0; i < DataLength / DataTypeLength; i++)
	{
		__m256 vect_in1 = _mm256_setr_ps(b[8 * i], b[8 * i + 1], b[8 * i + 2], b[8 * i + 3], b[8 * i + 4], b[8 * i + 5], b[8 * i + 6], b[8 * i + 7]);
		__m256 vect_in2 = _mm256_setr_ps(c[8 * i], c[8 * i + 1], c[8 * i + 2], c[8 * i + 3], c[8 * i + 4], c[8 * i + 5], c[8 * i + 6], c[8 * i + 7]);
		result[i] = _mm256_xor_ps(vect_in1, vect_in2);
	}
	s_elapsed = (dsecnd() - s_initial);
	printf(" completed with == \n == at %.5f milliseconds == \n\n", (s_elapsed * 1000));
	
	/*//debugging if DataTypeLength= 8 
	for (int j = 0; j < DataLength / DataTypeLength; j++)
	{
		float * ptr = (float *)& result[j];
		for (int i = 0; i < DataTypeLength; i++)
		{
			printf("%f  ", ptr[i]);
		}
		printf("\n\n");
	}
	*/
	return 0;
}

