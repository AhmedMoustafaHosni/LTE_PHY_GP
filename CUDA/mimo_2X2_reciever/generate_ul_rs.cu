/*
% Function:    generate_ul_rs
% Description: Generates LTE base reference signal sequence for the UL
% Inputs:      N_s                      - Slot number within a radio frame
%              N_id_cell                - Physical layer cell identity
%              chan_type                - pusch or pucch
%              delta_ss                 - Configurable portion of the sequence-shift pattern for PUSCH (sib2 groupAssignmentPUSCH)
%              group_hopping_enabled    - Boolean value determining if group hopping is enabled (sib2 groupHoppingEnabled)
%              sequence_hopping_enabled - Boolean value determining if sequence hopping is enabled (sib2 sequenceHoppingEnabled)
%              alpha                    - Cyclic shift
%              N_prbs                   - Number of PRBs used for the uplink grant
% Outputs:     ref_signal				- Base reference signal
By: Mohammed Mostafa
*/

#include "generate_ul_rs.cuh"
#include "generate_psuedo_random_seq.cuh"

__global__ void calculate_x_q(int q, int N_zc_rs, cufftComplex* x_q_d) {

	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (x_idx >= N_zc_rs)
		return;

	x_q_d[x_idx].x = cos(-PI*q*x_idx*(x_idx + 1) / N_zc_rs);
	x_q_d[x_idx].y = sin(-PI*q*x_idx*(x_idx + 1) / N_zc_rs);
}

__global__ void calculate_ref_sig_case1(cufftComplex* x_q_d, int N_zc_rs, float alpha, int M_sc_rs, cufftComplex* ref_signal_d) {

	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (x_idx >= M_sc_rs)
		return;

	//calculate r_bar_u_v and r_u_v (ref signal)
	ref_signal_d[x_idx].x = x_q_d[(x_idx % N_zc_rs)].x * cos(alpha*x_idx) - x_q_d[(x_idx % N_zc_rs)].y * sin(alpha*x_idx);
	ref_signal_d[x_idx].y = x_q_d[(x_idx % N_zc_rs)].x * sin(alpha*x_idx) + x_q_d[(x_idx % N_zc_rs)].y * cos(alpha*x_idx);
}

__global__ void calculate_ref_sig_case2(int u, float alpha, int M_sc_rs, const char* PHI_1_d, cufftComplex* ref_signal_d) {

	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (x_idx >= M_sc_rs)
		return;

	//calculate r_bar_u_v and r_u_v (ref signal)
	ref_signal_d[x_idx].x = cos(PHI_1_d[u*12 + x_idx] * PI/4) * cos(alpha*x_idx) - sin(PHI_1_d[u*12 + x_idx] * PI/4) * sin(alpha*x_idx);
	ref_signal_d[x_idx].y = cos(PHI_1_d[u*12 + x_idx] * PI/4) * sin(alpha*x_idx) + sin(PHI_1_d[u*12 + x_idx] * PI/4) * cos(alpha*x_idx);
}

__global__ void calculate_ref_sig_case3(int u, float alpha, int M_sc_rs, const char* PHI_2_d, cufftComplex* ref_signal_d) {

	int x_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (x_idx >= M_sc_rs)
		return;

	//calculate r_bar_u_v and r_u_v (ref signal)
	ref_signal_d[x_idx].x = cos(PHI_2_d[u * 12 + x_idx] * PI / 4) * cos(alpha*x_idx) - sin(PHI_2_d[u * 12 + x_idx] * PI / 4) * sin(alpha*x_idx);
	ref_signal_d[x_idx].y = cos(PHI_2_d[u * 12 + x_idx] * PI / 4) * sin(alpha*x_idx) + sin(PHI_2_d[u * 12 + x_idx] * PI / 4) * cos(alpha*x_idx);
}

void generate_ul_rs(int N_s, int N_id_cell, char* chan_type, int delta_ss, bool group_hopping_enabled, bool sequence_hopping_enabled, float alpha, int N_prbs, cufftComplex** ref_signal_d, cufftComplex* x_q_d)
{
	//Calculate M_sc_rs
	int M_sc_rs = N_prbs*N_sc_rb;

	//Determine N_zc_rs
	int N_zc_rs = prime_nums[N_prbs - 1];

	//Determine f_ss
	int f_ss;
	if (strcmp("pusch", chan_type))
		f_ss = ((N_id_cell % 30) + delta_ss) % 30;
	else 
		f_ss = N_id_cell % 30;

	//Determine u
	int u;
	if (1 == group_hopping_enabled)
	{
		Byte* c = (Byte*)malloc(sizeof(Byte) * 160);
		generate_psuedo_random_seq(&c, 160, 0, 0, floor(N_id_cell / 30));   //added c_init in N_id_cell according to ahmed nour
		int	f_gh =  c[8 * N_s + 0] + c[8 * N_s + 1] * 2 + c[8 * N_s + 2] * 4 + c[8 * N_s + 3] * 8 + c[8 * N_s + 4] * 16 + c[8 * N_s + 5] * 32 + c[8 * N_s + 6] * 64 + c[8 * N_s + 7] * 128;
		f_gh = f_gh % 30;
		u = (f_gh + f_ss) % 30;
	}
	else
		u = f_ss % 30;

	//Determine v
	int v;
	if (M_sc_rs < 6 * N_sc_rb)
		v = 0;
	else
	{
		if (0 == group_hopping_enabled && 1 == sequence_hopping_enabled)
		{
			Byte* c = (Byte*)malloc(sizeof(Byte) * 160);
			generate_psuedo_random_seq(&c, 20, 0, 0, floor(N_id_cell / 30) * 32);   //added c_init in N_id_cell according to ahmed nour
			v = c[N_s];
		}
		else
			v = 0;
	}
	
	//Determine r_bar_u_v   &&   r_u_v (Calculate reference signal)
	if (M_sc_rs >= 3 * N_sc_rb)
	{
		float q_bar = N_zc_rs*(u + 1) / (float)31;
		int q = floor(q_bar + 0.5) + pow(v*(-1) , floor(2*q_bar));

		calculate_x_q << < 2 , 1024 >> >(q, N_zc_rs, x_q_d);
		calculate_ref_sig_case1 << < 2, 1024 >> >(x_q_d, N_zc_rs, alpha, M_sc_rs, *ref_signal_d);
	}
	else if (M_sc_rs == N_sc_rb)
	{
		calculate_ref_sig_case2 << < 2, 1024 >> >(u, alpha, M_sc_rs, PHI_1_d, *ref_signal_d);
	}
	else
	{
		calculate_ref_sig_case3 << < 2, 1024 >> >(u, alpha, M_sc_rs, PHI_2_d, *ref_signal_d);
	}

}