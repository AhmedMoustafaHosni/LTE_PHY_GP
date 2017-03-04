/* This APP is used to test SCFDMA Modulator*/
// BY Khaled Ahmed Ali 

#include "Intel_Header.h"
using namespace std;

int main()
{
	/*inputs*/
	int M_pusch_rb = 100;
	_MKL_Complex8 input_subframe[14][1200]; // 14 symboles in one subfram - 1200 subcarrier assigned for this single user
	//just for test
	input_subframe[:][0:1200].real = 1;
	input_subframe[:][:].imag = 0;
	_MKL_Complex8 pusch_bb[30720]; // the output

	/*1st Call*/
	SC_FDMA_mod(pusch_bb, M_pusch_rb, input_subframe);


	/*2nd Call*/
	double s_initial = 0, s_elapsed = 0;
	s_initial = dsecnd();  // sample time
	SC_FDMA_mod(pusch_bb, M_pusch_rb, input_subframe);
	s_elapsed = (dsecnd() - s_initial);
	printf(" completed task1 with == \n == at %.5f milliseconds == \n\n", (s_elapsed * 1000));

	/* verify the output with matlab*/
	/*
	1- take the output from the file and put it ito matlab (copy-past)
	2- diff = x - pusch_bb;
	3- error = sum(diff) => that error can be reduced using Double Precision IFFT 
	*/
	FILE *fp;
	fp = fopen("Output3.txt", "w");
	fprintf(fp, "x= ");
	fprintf(fp, "[");
	fprintf(fp, "%lf + 1i * %lf ", pusch_bb[0:30720].real, pusch_bb[0:30720].imag);
	fprintf(fp, "]");
	fprintf(fp, ";");
	
	return 0;
}