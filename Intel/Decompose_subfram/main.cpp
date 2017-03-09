#include "Header.h"


int main()
{
	/*Data come from Transform Precoder*/
	MKL_Complex8 dmrs_1[1200];
	MKL_Complex8 dmrs_2[1200];
	MKL_Complex8 data[1200 * 14];
	dmrs_1[:].real = 0;
	dmrs_2[:].real = 0;
	data[:].real = 1;
	dmrs_1[:].imag = 0;
	dmrs_2[:].imag = 0;
	data[:].imag = 0;
	int RB = 100;

	/*Call The function*/
	MKL_Complex8 ** gride_out = NULL;
	double s_initial = 0, s_elapsed = 0;
	s_initial = dsecnd();  // sample time
	gride_out = compose_subframe(data, dmrs_1, dmrs_2, RB);
	s_elapsed = (dsecnd() - s_initial);
	printf(" completed task1 with == \n == at %.5f milliseconds == \n\n", (s_elapsed * 1000));

	/*Call decompose*/
	s_initial = 0, s_elapsed = 0;
	s_initial = dsecnd();  // sample time
	decompose_subframe(gride_out,data, dmrs_1, dmrs_2, RB);
	s_elapsed = (dsecnd() - s_initial);
	printf(" completed task2 with == \n == at %.5f milliseconds == \n\n", (s_elapsed * 1000));

	FILE * fp;
	fp = fopen("output.txt", "w");
	fprintf(fp, "REAL = %f    IMAG = %f\n", gride_out[0:13][0:1200].real, gride_out[0:13][0:1200].imag);
	fclose(fp);

	FILE * fp2;
	fp2 = fopen("output2.txt", "w");
	fprintf(fp2, "REAL = %f    IMAG = %f\n", data[0:14400].real, data[0:14400].imag);
	fclose(fp2);
	return 0;
}