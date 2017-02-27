#include "Intel_Header.h"

static int APP1(void);
//static int APP2(void);
//static int APP3(void);

int main()
{
	/*APP1*/
	if (APP1() != 0)
		printf("error in APP1\n");
	return 0;
}