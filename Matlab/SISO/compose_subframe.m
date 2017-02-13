% Function:    compose_subframe
% Description: compose the subframe by multiplexing the dmrs signal and data 
% Inputs:      data                  - complex data to be sent in subframe
%              dmrs_1                - demodulation reference signal number 1
%              dmrs_2                - demodulation reference signal number 2
%              M_pusch_rb            - number of resource blocks assigned to the ue
% Outputs:     subframe              - the subframe with data of each ofdm symbol in each row
% edit 27/1/2017
%by Ahmed Moustafa

function subframe = compose_subframe(data, dmrs_1, dmrs_2, M_pusch_rb)
 N_sc_rb   = 12;      %% number of subcarriers in each resource block
 M_pusch_sc = N_sc_rb * M_pusch_rb;

 idx = 0;
 N_symbs_per_slot=7;
 for L = 0:(N_symbs_per_slot*2)-1
	 if(3 == L)
		 for k=0:M_pusch_sc-1 
			 % DMRS
			 subframe(L+1,k+1) = dmrs_1(k+1);
		 end
	 elseif(10 == L)
		 for k=0:M_pusch_sc-1 
			 % DMRS
			 subframe(L+1,k+1) = dmrs_2(k+1);
		 end
	 else
		 for k=0:M_pusch_sc-1
			 % PUSCH
			 subframe(L+1,k+1) = data(idx+1);
			 idx                  = idx + 1;
		 end
	 end
 end
 
 
end