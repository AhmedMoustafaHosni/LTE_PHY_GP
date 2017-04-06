/*******************************************************************************
* Function:    generate_dmrs
* Description: Generates LTE demodulation reference signal for PUSCH
* Inputs:      N_subframe         - Subframe number within a radio frame
*			   cell_id            - Physical layer cell identity
*              delta_ss           - Configurable portion by higher layer of the sequence-shift pattern for PUSCH --> {0,1,...,29}
*			   cyclic_shift       - cyclic shift to apply to base reference signal --> {0,1,2,...,7}
*		       cyclic_shift_dci   - Scheduled cyclic shift to apply to base reference signal --> {0,1,2,...,7}
*			   N_rb               - Number of RBs assigned for the UE
*
* Outputs:     dmrs_1        - Demodulation reference signal for the 4th symbol
*			   dmrs_2        - Demodulation reference signal for the 11th symbol
*
* Max timing (for 86400 bits and 64QAM): 0.083 msecs 
* 
*
* by: Mohammed Osama
********************************************************************************/

#include "Intel_siso.h"

void generate_dmrs(unsigned char N_suframe, unsigned int cell_id, unsigned char delta_ss, unsigned char cyclic_shift, unsigned char cyclic_shift_dci, unsigned char N_rb, MKL_Complex8*& dmrs_1, MKL_Complex8*& dmrs_2)
{
	// Only dealing with normal cp
	char N_symbols_up = 7;

	/******************************  compute alpha : alpha = 2 * pi * ncs /12  *******************/

	//ncs = ( nDMRS_1 + nDMRS_2 + nPRS_ns ) mod12
	// ns is the number of slot

	// available nDMRS according to the desired cyclic shift
	char N_1_DMRS[] = { 0, 2, 3, 4, 6, 8, 9, 10 };
	char N_2_DMRS[] = { 0, 6, 3, 4, 2, 8, 10, 9 };

	char nDMRS_1  = N_1_DMRS[cyclic_shift];
	char nDMRS_2 = N_2_DMRS[cyclic_shift_dci];

	// the number of the first slot in the subframe
	unsigned char ns = N_suframe * 2;

	int nPRS_ns_1 = 0, nPRS_ns_2 = 0;

	// generate seudo random sequence
	char f_ss_pusch = ((cell_id % 30) + delta_ss) % 30;
	
	// original equation: c_init = floor(cell_id/30) * 32 + f_ss_pusch
	// convert floor(cell_id / 30) ---> cell_id/30, don't approximate 30 with 32
	unsigned int c_init = cell_id / 30 * 32 + f_ss_pusch;


	/***** call the seudo random sequence function ****/

	unsigned short* c = pseudo_random_sequence_gen(c_init, 1120);
	
	/**************************************************/
	
	// comment when the pseudo random sequence is finished
	//char c[1120] = { 0 };
	
	/*for (int n = 0; n <= 7; n++) {
		nPRS_ns_1 = nPRS_ns_1 + c[8 * N_symbols_up*ns + n] * pow (2,n);
		nPRS_ns_2 = nPRS_ns_2 + c[8 * N_symbols_up*(ns + 1) + n] * pow(2,n);
	}*/

	// loop unrolling
	int tmp1 = 8 * N_symbols_up * ns;
	int tmp2 = 8 * N_symbols_up * (ns + 1);
	nPRS_ns_1 = c[tmp1] + c[tmp1 + 1] * 2 + c[tmp1 + 2] * 4 + c[tmp1 + 3] * 8 + c[tmp1 + 4] * 16 + c[tmp1 + 5] * 32 + c[tmp1 + 6] * 64 + c[tmp1 + 7] * 128;
	nPRS_ns_2 = c[tmp2] + c[tmp2 + 1] * 2 + c[tmp2 + 2] * 4 + c[tmp2 + 3] * 8 + c[tmp2 + 4] * 16 + c[tmp2 + 5] * 32 + c[tmp2 + 6] * 64 + c[tmp2 + 7] * 128;
	
	int ncs_1 = (nDMRS_1 + nDMRS_2 + nPRS_ns_1)% 12;
	int ncs_2 = (nDMRS_1 + nDMRS_2 + nPRS_ns_2)% 12;

	double alpha_1 = M_PI_6 *ncs_1;   // alpha = pi/6 * ncs
	double alpha_2 = M_PI_6 *ncs_2;

	/*********************************************************/

	/************************************************* generate the "base reference signal"  *****************************************/

	// 2 possibilities :  1) UE is assigned less than 3 RBs ---> base = exp(j*phi(n) * pi /4)
	//					  2) UE is assigned more than 3 RBs ---> base = x_q(n % N_RS_ZC)

	MKL_Complex8 r_bar_u_v[M_RS_SC_MAX];

	int M_rs_sc = N_rb * 12;  // length of reference signal

	// phi(n) when 1 resource block assigned for the user 
	char PHI_1[30][12] = { { -1, 1, 3,-3, 3, 3, 1, 1, 3, 1,-3, 3 },
							 {  1, 1, 3, 3, 3,-1, 1,-3,-3, 1,-3, 3 },
							 {  1, 1,-3,-3,-3,-1,-3,-3, 1,-3, 1,-1 },
							 { -1, 1, 1, 1, 1,-1,-3,-3, 1,-3, 3,-1 },
							 { -1, 3, 1,-1, 1,-1,-3,-1, 1,-1, 1, 3 },
							 {  1,-3, 3,-1,-1, 1, 1,-1,-1, 3,-3, 1 },
							 { -1, 3,-3,-3,-3, 3, 1,-1, 3, 3,-3, 1 },
							 { -3,-1,-1,-1, 1,-3, 3,-1, 1,-3, 3, 1 },
							  { 1,-3, 3, 1,-1,-1,-1, 1, 1, 3,-1, 1 },
							  { 1,-3,-1, 3, 3,-1,-3, 1, 1, 1, 1, 1 },
							  {-1, 3,-1, 1, 1,-3,-3,-1,-3,-3, 3,-1 },
							  { 3, 1,-1,-1, 3, 3,-3, 1, 3, 1, 3, 3 },
							  { 1,-3, 1, 1,-3, 1, 1, 1,-3,-3,-3, 1 },
							  { 3, 3,-3, 3,-3, 1, 1, 3,-1,-3, 3, 3 },
							  {-3, 1,-1,-3,-1, 3, 1, 3, 3, 3,-1, 1 },
							  { 3,-1, 1,-3,-1,-1, 1, 1, 3, 1,-1,-3 },
							  { 1, 3, 1,-1, 1, 3, 3, 3,-1,-1, 3,-1 },
							  {-3, 1, 1, 3,-3, 3,-3,-3, 3, 1, 3,-1 },
							  {-3, 3, 1, 1,-3, 1,-3,-3,-1,-1, 1,-3 },
							  {-1, 3, 1, 3, 1,-1,-1, 3,-3,-1,-3,-1 },
							  {-1,-3, 1, 1, 1, 1, 3, 1,-1, 1,-3,-1 },
							  {-1, 3,-1, 1,-3,-3,-3,-3,-3, 1,-1,-3 },
							  { 1, 1,-3,-3,-3,-3,-1, 3,-3, 1,-3, 3 },
							  { 1, 1,-1,-3,-1,-3, 1,-1, 1, 3,-1, 1 },
							  { 1, 1, 3, 1, 3, 3,-1, 1,-1,-3,-3, 1 },
							  { 1,-3, 3, 3, 1, 3, 3, 1,-3,-1,-1, 3 },
							  { 1, 3,-3,-3, 3,-3, 1,-1,-1, 3,-1,-3 },
							  {-3,-1,-3,-1,-3, 3, 1,-1, 1, 3,-3,-3 },
							  {-1, 3,-3, 3,-1, 3, 3,-3, 3, 3,-1,-1 },
							  { 3,-3,-3,-1,-1,-3,-1, 3,-3, 3, 1,-1 } };

	// phi(n) when 2 resource block assigned for the user
	char PHI_2[30][24] = {   { -1, 3, 1,-3, 3,-1, 1, 3,-3, 3, 1, 3,-3, 3, 1, 1,-1, 1, 3,-3, 3,-3,-1,-3 },
							 { -3, 3,-3,-3,-3, 1,-3,-3, 3,-1, 1, 1, 1, 3, 1,-1, 3,-3,-3, 1, 3, 1, 1,-3 },
							 {  3,-1, 3, 3, 1, 1,-3, 3, 3, 3, 3, 1,-1, 3,-1, 1, 1,-1,-3,-1,-1, 1, 3, 3 },
							 { -1,-3, 1, 1, 3,-3, 1, 1,-3,-1,-1, 1, 3, 1, 3, 1,-1, 3, 1, 1,-3,-1,-3,-1 },
							 { -1,-1,-1,-3,-3,-1, 1, 1, 3, 3,-1, 3,-1, 1,-1,-3, 1,-1,-3,-3, 1,-3,-1,-1 },
							 { -3, 1, 1, 3,-1, 1, 3, 1,-3, 1,-3, 1, 1,-1,-1, 3,-1,-3, 3,-3,-3,-3, 1, 1 },
							 {  1, 1,-1,-1, 3,-3,-3, 3,-3, 1,-1,-1, 1,-1, 1, 1,-1,-3,-1, 1,-1, 3,-1,-3 },
							 { -3, 3, 3,-1,-1,-3,-1, 3, 1, 3, 1, 3, 1, 1,-1, 3, 1,-1, 1, 3,-3,-1,-1, 1 },
							 { -3, 1, 3,-3, 1,-1,-3, 3,-3, 3,-1,-1,-1,-1, 1,-3,-3,-3, 1,-3,-3,-3, 1,-3 },
							 {  1, 1,-3, 3, 3,-1,-3,-1, 3,-3, 3, 3, 3,-1, 1, 1,-3, 1,-1, 1, 1,-3, 1, 1 },
							 { -1, 1,-3,-3, 3,-1, 3,-1,-1,-3,-3,-3,-1,-3,-3, 1,-1, 1, 3, 3,-1, 1,-1, 3 },
							 {  1, 3, 3,-3,-3, 1, 3, 1,-1,-3,-3,-3, 3, 3,-3, 3, 3,-1,-3, 3,-1, 1,-3, 1 },
							 {  1, 3, 3, 1, 1, 1,-1,-1, 1,-3, 3,-1, 1, 1,-3, 3, 3,-1,-3, 3,-3,-1,-3,-1 },
							 {  3,-1,-1,-1,-1,-3,-1, 3, 3, 1,-1, 1, 3, 3, 3,-1, 1, 1,-3, 1, 3,-1,-3, 3 },
							 { -3,-3, 3, 1, 3, 1,-3, 3, 1, 3, 1, 1, 3, 3,-1,-1,-3, 1,-3,-1, 3, 1, 1, 3 },
							 { -1,-1, 1,-3, 1, 3,-3, 1,-1,-3,-1, 3, 1, 3, 1,-1,-3,-3,-1,-1,-3,-3,-3,-1 },
							 { -1,-3, 3,-1,-1,-1,-1, 1, 1,-3, 3, 1, 3, 3, 1,-1, 1,-3, 1,-3, 1, 1,-3,-1 },
							 {  1, 3,-1, 3, 3,-1,-3, 1,-1,-3, 3, 3, 3,-1, 1, 1, 3,-1,-3,-1, 3,-1,-1,-1 },
							 {  1, 1, 1, 1, 1,-1, 3,-1,-3, 1, 1, 3,-3, 1,-3,-1, 1, 1,-3,-3, 3, 1, 1,-3 },
							 {  1, 3, 3, 1,-1,-3, 3,-1, 3, 3, 3,-3, 1,-1, 1,-1,-3,-1, 1, 3,-1, 3,-3,-3 },
							 { -1,-3, 3,-3,-3,-3,-1,-1,-3,-1,-3, 3, 1, 3,-3,-1, 3,-1, 1,-1, 3,-3, 1,-1 },
							 { -3,-3, 1, 1,-1, 1,-1, 1,-1, 3, 1,-3,-1, 1,-1, 1,-1,-1, 3, 3,-3,-1, 1,-3 },
							 { -3,-1,-3, 3, 1,-1,-3,-1,-3,-3, 3,-3, 3,-3,-1, 1, 3, 1,-3, 1, 3, 3,-1,-3 },
							 { -1,-1,-1,-1, 3, 3, 3, 1, 3, 3,-3, 1, 3,-1, 3,-1, 3, 3,-3, 3, 1,-1, 3, 3 },
							 {  1,-1, 3, 3,-1,-3, 3,-3,-1,-1, 3,-1, 3,-1,-1, 1, 1, 1, 1,-1,-1,-3,-1, 3 },
							 {  1,-1, 1,-1, 3,-1, 3, 1, 1,-1,-1,-3, 1, 1,-3, 1, 3,-3, 1, 1,-3,-3,-1,-1 },
							 { -3,-1, 1, 3, 1, 1,-3,-1,-1,-3, 3,-3, 3, 1,-3, 3,-3, 1,-1, 1,-3, 1, 1, 1 },
							 { -1,-3, 3, 3, 1, 1, 3,-1,-3,-1,-1,-1, 3, 1,-3,-3,-1, 3,-3,-1,-3,-1,-3,-1 },
							 { -1,-3,-1,-1, 1,-3,-1,-1, 1,-1,-3, 1, 1,-3, 1,-3,-3, 3, 1, 1,-1, 3,-1,-1 },
							 {  1, 1,-1,-1,-3,-1, 3,-1, 3,-1, 1, 3, 1,-1, 3, 1, 3,-3,-3, 1,-1,-1, 1, 3 }};

	// when more than 3 RB assigned for the user
	// adding only the used prime numbers
	int prime_nums[110] = { 11, 23, 31, 47, 59, 71, 83, 89, 107, 113, 131, 139, 151, 167, 179, 191, 199, 211, 227, 239, 251, 263, 271, 283, 293, 311, 317, 331, 347, 359, 367, 383, 389, 401, 419, 431, 443, 449, 467, 479, 491, 503, 509, 523, 523, 547, 563, 571, 587, 599, 607, 619, 631, 647, 659, 661, 683, 691, 701, 719, 727, 743, 751, 761, 773, 787, 797, 811, 827, 839, 839, 863, 863, 887, 887, 911, 919, 929, 947, 953, 971, 983, 991, 997, 1019, 1031, 1039, 1051, 1063, 1069, 1091, 1103, 1109, 1123, 1129, 1151, 1163, 1171, 1187, 1193, 1201, 1223, 1231, 1237, 1259, 1259, 1283, 1291, 1307, 1319};
	
	//u = f_ss_pusch % 30 which is equal to f_ss_pusch
	int u = f_ss_pusch;

	if (N_rb >= 3)
	{
		/** 2nd possibility **/

		// zadoof chu length sequence is the largest nearest prime number nearest to number of sub-carriers in N_rb 
		int N_zc_rs = prime_nums[N_rb - 1];	
 
		float q_bar = N_zc_rs * (u + 1) / 31.0;

		// assume no sequence and group hopping --> v=0
		float q = floor(q_bar + 0.5);
		const double power = - M_PI * q / (float) N_zc_rs;

		// define the maximum size of array with the maximum zadoff chu length
		//MKL_Complex8 x_q [N_ZC_RS_MAX];

		// generate the base reference signal
		// loop fusion of x_q and r_bar_u_v
		for (int m = 0; m < N_zc_rs; m++)
		{
			double tmp = m * power * (m + 1);
			r_bar_u_v[m].real = (float) cos(tmp);
			r_bar_u_v[m].imag = (float) sin(tmp);
		}

		// the remaining of the base reference signal
		for (int m = N_zc_rs; m < M_rs_sc; m++)
		{
			r_bar_u_v[m] = r_bar_u_v[m % N_zc_rs];
			//r_bar_u_v[m] = x_q[m % N_zc_rs];
		}

	}	
	else if (N_rb == 1) {
		/** 1st possibility **/
		/*for (int m = 0; m < M_rs_sc; m++)
		{
			r_bar_u_v[m].real = cos(M_PI_4 * PHI_1[u][m]);
			r_bar_u_v[m].imag = ;
		}*/
		r_bar_u_v[0:M_rs_sc].real = cos(M_PI_4 * PHI_1[u][0:M_rs_sc]);
		r_bar_u_v[0:M_rs_sc].imag = sin(M_PI_4 * PHI_1[u][0:M_rs_sc]);
	}
	else {
		r_bar_u_v[0:M_rs_sc].real = cos(M_PI_4 * PHI_2[u][0:M_rs_sc]);
		r_bar_u_v[0:M_rs_sc].imag = sin(M_PI_4 * PHI_2[u][0:M_rs_sc]);
	}

	/************************************  generate the two dmrs signals *****************************************/
	
	dmrs_1 = (MKL_Complex8 *) malloc(M_rs_sc * sizeof(MKL_Complex8));
	dmrs_2 = (MKL_Complex8 *) malloc(M_rs_sc * sizeof(MKL_Complex8));
	
	// dmrs = exp(j*alpha*n) * r_bar_u_v 
	for (int i = 0; i < M_rs_sc; i++)
	{
		// (a+bj)(c+dj) = (ac - bd) + j(ad + bc)
		dmrs_1[i].real = cos(alpha_1 * i) * r_bar_u_v[i].real - sin(alpha_1 * i) * r_bar_u_v[i].imag;
		dmrs_1[i].imag = cos(alpha_1 * i) * r_bar_u_v[i].imag + sin(alpha_1 * i) * r_bar_u_v[i].real;
		dmrs_2[i].real = cos(alpha_2 * i) * r_bar_u_v[i].real - sin(alpha_2 * i) * r_bar_u_v[i].imag;
		dmrs_2[i].imag = cos(alpha_2 * i) * r_bar_u_v[i].imag + sin(alpha_2 * i) * r_bar_u_v[i].real;	
	}

}