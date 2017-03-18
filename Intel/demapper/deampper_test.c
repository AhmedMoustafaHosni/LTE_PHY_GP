

#include <stdio.h>
#include <mkl.h>
#include "demapper.h"

void print_array_of_float(float * array, int lenght);
void print_for_test(FILE * fileID, float * array, int lenght);


int main(void)
{
	float bits_bpsk[2];
	MKL_Complex8 symbols_bpsk[2];

	float bits_qpsk[8];
	MKL_Complex8 symbol_qpsk[4];

	float bits_16qam[64];
	MKL_Complex8 symbol_16qam[16];

	float bits_64qam[384];
	MKL_Complex8 symbol_64qam[64];

	symbols_bpsk[0].real = symbols_bpsk[0].imag = 0.71;
	symbols_bpsk[1].real = symbols_bpsk[1].imag = -0.71;

	symbol_qpsk[0].real = 0.710000;
	symbol_qpsk[0].imag = 0.710000;
	symbol_qpsk[1].real = 0.710000;
	symbol_qpsk[1].imag = -0.710000;
	symbol_qpsk[2].real = -0.710000;
	symbol_qpsk[2].imag = 0.710000;
	symbol_qpsk[3].real = -0.710000;
	symbol_qpsk[3].imag = -0.710000;



	symbol_16qam[0].real = 0.320000;
	symbol_16qam[0].imag = 0.320000;
	symbol_16qam[1].real = -0.320000;
	symbol_16qam[1].imag = 0.320000;
	symbol_16qam[2].real = 0.320000;
	symbol_16qam[2].imag = -0.320000;
	symbol_16qam[3].real = -0.320000;
	symbol_16qam[3].imag = -0.320000;
	symbol_16qam[4].real = 0.950000;
	symbol_16qam[4].imag = 0.320000;
	symbol_16qam[5].real = -0.950000;
	symbol_16qam[5].imag = 0.320000;
	symbol_16qam[6].real = 0.950000;
	symbol_16qam[6].imag = -0.320000;
	symbol_16qam[7].real = -0.950000;
	symbol_16qam[7].imag = -0.320000;
	symbol_16qam[8].real = 0.320000;
	symbol_16qam[8].imag = 0.950000;
	symbol_16qam[9].real = -0.320000;
	symbol_16qam[9].imag = 0.950000;
	symbol_16qam[10].real = 0.320000;
	symbol_16qam[10].imag = -0.950000;
	symbol_16qam[11].real = -0.320000;
	symbol_16qam[11].imag = -0.950000;
	symbol_16qam[12].real = 0.950000;
	symbol_16qam[12].imag = 0.950000;
	symbol_16qam[13].real = -0.950000;
	symbol_16qam[13].imag = 0.950000;
	symbol_16qam[14].real = 0.950000;
	symbol_16qam[14].imag = -0.950000;
	symbol_16qam[15].real = -0.950000;
	symbol_16qam[15].imag = -0.950000;



	symbol_64qam[0].real = 0.460000;
	symbol_64qam[0].imag = 0.460000;
	symbol_64qam[1].real = -0.460000;
	symbol_64qam[1].imag = 0.460000;
	symbol_64qam[2].real = 0.460000;
	symbol_64qam[2].imag = -0.460000;
	symbol_64qam[3].real = -0.460000;
	symbol_64qam[3].imag = -0.460000;
	symbol_64qam[4].real = 0.770000;
	symbol_64qam[4].imag = 0.460000;
	symbol_64qam[5].real = -0.770000;
	symbol_64qam[5].imag = 0.460000;
	symbol_64qam[6].real = 0.770000;
	symbol_64qam[6].imag = -0.460000;
	symbol_64qam[7].real = -0.770000;
	symbol_64qam[7].imag = -0.460000;
	symbol_64qam[8].real = 0.460000;
	symbol_64qam[8].imag = 0.770000;
	symbol_64qam[9].real = -0.460000;
	symbol_64qam[9].imag = 0.770000;
	symbol_64qam[10].real = 0.460000;
	symbol_64qam[10].imag = -0.770000;
	symbol_64qam[11].real = -0.460000;
	symbol_64qam[11].imag = -0.770000;
	symbol_64qam[12].real = 0.770000;
	symbol_64qam[12].imag = 0.770000;
	symbol_64qam[13].real = -0.770000;
	symbol_64qam[13].imag = 0.770000;
	symbol_64qam[14].real = 0.770000;
	symbol_64qam[14].imag = -0.770000;
	symbol_64qam[15].real = -0.770000;
	symbol_64qam[15].imag = -0.770000;
	symbol_64qam[16].real = 0.150000;
	symbol_64qam[16].imag = 0.460000;
	symbol_64qam[17].real = -0.150000;
	symbol_64qam[17].imag = 0.460000;
	symbol_64qam[18].real = 0.150000;
	symbol_64qam[18].imag = -0.460000;
	symbol_64qam[19].real = -0.150000;
	symbol_64qam[19].imag = -0.460000;
	symbol_64qam[20].real = 1.080000;
	symbol_64qam[20].imag = 0.460000;
	symbol_64qam[21].real = -1.080000;
	symbol_64qam[21].imag = 0.460000;
	symbol_64qam[22].real = 1.080000;
	symbol_64qam[22].imag = -0.460000;
	symbol_64qam[23].real = -1.080000;
	symbol_64qam[23].imag = -0.460000;
	symbol_64qam[24].real = 0.150000;
	symbol_64qam[24].imag = 0.770000;
	symbol_64qam[25].real = -0.150000;
	symbol_64qam[25].imag = 0.770000;
	symbol_64qam[26].real = 0.150000;
	symbol_64qam[26].imag = -0.770000;
	symbol_64qam[27].real = -0.150000;
	symbol_64qam[27].imag = -0.770000;
	symbol_64qam[28].real = 1.080000;
	symbol_64qam[28].imag = 0.770000;
	symbol_64qam[29].real = -1.080000;
	symbol_64qam[29].imag = 0.770000;
	symbol_64qam[30].real = 1.080000;
	symbol_64qam[30].imag = -0.770000;
	symbol_64qam[31].real = -1.080000;
	symbol_64qam[31].imag = -0.770000;
	symbol_64qam[32].real = 0.460000;
	symbol_64qam[32].imag = 0.150000;
	symbol_64qam[33].real = -0.460000;
	symbol_64qam[33].imag = 0.150000;
	symbol_64qam[34].real = 0.460000;
	symbol_64qam[34].imag = -0.150000;
	symbol_64qam[35].real = -0.460000;
	symbol_64qam[35].imag = -0.150000;
	symbol_64qam[36].real = 0.770000;
	symbol_64qam[36].imag = 0.150000;
	symbol_64qam[37].real = -0.770000;
	symbol_64qam[37].imag = 0.150000;
	symbol_64qam[38].real = 0.770000;
	symbol_64qam[38].imag = -0.150000;
	symbol_64qam[39].real = -0.770000;
	symbol_64qam[39].imag = -0.150000;
	symbol_64qam[40].real = 0.460000;
	symbol_64qam[40].imag = 1.080000;
	symbol_64qam[41].real = -0.460000;
	symbol_64qam[41].imag = 1.080000;
	symbol_64qam[42].real = 0.460000;
	symbol_64qam[42].imag = -1.080000;
	symbol_64qam[43].real = -0.460000;
	symbol_64qam[43].imag = -1.080000;
	symbol_64qam[44].real = 0.770000;
	symbol_64qam[44].imag = 1.080000;
	symbol_64qam[45].real = -0.770000;
	symbol_64qam[45].imag = 1.080000;
	symbol_64qam[46].real = 0.770000;
	symbol_64qam[46].imag = -1.080000;
	symbol_64qam[47].real = -0.770000;
	symbol_64qam[47].imag = -1.080000;
	symbol_64qam[48].real = 0.150000;
	symbol_64qam[48].imag = 0.150000;
	symbol_64qam[49].real = -0.150000;
	symbol_64qam[49].imag = 0.150000;
	symbol_64qam[50].real = 0.150000;
	symbol_64qam[50].imag = -0.150000;
	symbol_64qam[51].real = -0.150000;
	symbol_64qam[51].imag = -0.150000;
	symbol_64qam[52].real = 1.080000;
	symbol_64qam[52].imag = 0.150000;
	symbol_64qam[53].real = -1.080000;
	symbol_64qam[53].imag = 0.150000;
	symbol_64qam[54].real = 1.080000;
	symbol_64qam[54].imag = -0.150000;
	symbol_64qam[55].real = -1.080000;
	symbol_64qam[55].imag = -0.150000;
	symbol_64qam[56].real = 0.150000;
	symbol_64qam[56].imag = 1.080000;
	symbol_64qam[57].real = -0.150000;
	symbol_64qam[57].imag = 1.080000;
	symbol_64qam[58].real = 0.150000;
	symbol_64qam[58].imag = -1.080000;
	symbol_64qam[59].real = -0.150000;
	symbol_64qam[59].imag = -1.080000;
	symbol_64qam[60].real = 1.080000;
	symbol_64qam[60].imag = 1.080000;
	symbol_64qam[61].real = -1.080000;
	symbol_64qam[61].imag = 1.080000;
	symbol_64qam[62].real = 1.080000;
	symbol_64qam[62].imag = -1.080000;
	symbol_64qam[63].real = -1.080000;
	symbol_64qam[63].imag = -1.080000;


	
	demapper(symbols_bpsk, 2, 2, bits_bpsk);
	demapper(symbol_qpsk, 4, 4, bits_qpsk);
	demapper(symbol_16qam, 16, 16, bits_16qam);
	demapper(symbol_64qam, 64, 64, bits_64qam);



	print_array_of_float(bits_bpsk, 2);
	printf("\n");
	print_array_of_float(bits_qpsk, 8);
	printf("\n");
	print_array_of_float(bits_16qam, 64);
	printf("\n");
	print_array_of_float(bits_64qam, 383);
	printf("\n");

	FILE * fileID;
	fileID = fopen("out.m", "w");


	print_for_test(fileID, bits_bpsk, 2);
	fprintf(fileID, "\n ");
	print_for_test(fileID, bits_qpsk, 8);
	fprintf(fileID, "\n ");
	fprintf(fileID, "result_16 =  ");
	print_for_test(fileID, bits_16qam, 64);
	fprintf(fileID, ";");
	fprintf(fileID, "\n ");
	fprintf(fileID, "result_64 =  ");
	print_for_test(fileID, bits_64qam, 384);
	fprintf(fileID, ";");
	fprintf(fileID, "\n ");
	fclose(fileID);

	int number_of_symbols = 43200;
	MKL_Complex8 symbols[43200];
	float bits[86400];
	for (int ii = 0; ii < number_of_symbols - 4; ii = ii + 4)
	{
		symbols[ii].real = 0.7;
		symbols[ii].imag = 0.7;
		symbols[ii + 1].real = 0.7;
		symbols[ii + 1].imag = -0.7;
		symbols[ii + 2].real = -0.7;
		symbols[ii + 2].imag = 0.7;
		symbols[ii + 3].real = -0.7;
		symbols[ii + 3].imag = -0.7;
	}
	double time_before = dsecnd();
	double execution_time;

	//while (1)
	//{
	time_before = dsecnd();
	demapper(symbols, number_of_symbols, 4, bits);
	execution_time = (dsecnd() - time_before);

	//print_array_of_float(bits, number_of_bits);
	printf("\n");
	printf("time = %f   milliseconds", execution_time*1000.0);
	printf("\n");

	//char myChar = getchar();
	//}
}






void print_array_of_float(float * array, int lenght)
{
	for (int i = 0; i < lenght; i++)
	{
		printf("%f\n", array[i]);
	}
}

void print_for_test(FILE * fileID, float * array, int lenght)
{

	fprintf(fileID, "[");
	for (int i = 0; i < lenght; i++)
	{
		fprintf(fileID,"%f ", array[i]);
	}
	fprintf(fileID, "]");

}

