/* mapper_main.cpp
*  test for mapper 
*/

#include "mapper.h"

using namespace std;

int main()
{
	// Intializing input array
	float bits[FRAME_LENGTH];
	// Intializing input array
	for (int i = 0; i < FRAME_LENGTH; i++)
	{
		bits[i] = 0;
	}

	// measure timing 
	double s_initial = 0, s_elapsed = 0;
	s_initial = dsecnd();

	// mapping bits to symbols
	// bits*-1.4142 + 0.7071 

	MKL_Complex8 symbols[FRAME_LENGTH / 2];
	mapper(bits, FRAME_LENGTH, symbols, MOD);

	s_elapsed = (dsecnd() - s_initial);

	//for (int i = 0; i < FRAME_LENGTH / 2; i++) {
	//	cout << symbols[i].real << " +j " << symbols[i].imag << endl;
	//	//printf("%.4f +j %.4f\n", symbols[i].real, symbols[i].imag);
	//}

	printf(" completed == \n"
		" == at %.5f milliseconds == \n\n", (s_elapsed * 1000));
	return 0;
}