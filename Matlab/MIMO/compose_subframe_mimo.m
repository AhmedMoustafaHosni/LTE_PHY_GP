% Function:    compose_subframe_mimo
% Description: compose the subframe by multiplexing the dmrs signal and data
% Inputs:      data                  - complex data to be sent in subframe
%			   matrix of size (P * M_symbols)
%              dmrs_1                - demodulation reference signal number
%              1 matrix of size (P * M)
%              dmrs_2                - demodulation reference signal number
%              2  matrix of size (P * M)
% where P is the antenna ports
%       M is the reference symbols
%              M_pusch_rb            - number of resource blocks assigned to the ue
% Outputs:     subframe              - the subframes for each antenna with
% 3D matrix with dim (M_pusch_sc_ * 14 (M_symbs) * P)
%									   
% edit 27/1/2017
%by Ahmed Moustafa
% Add MIMO by Mohammed Osama

function subframe = compose_subframe_mimo(data, dmrs_1, dmrs_2, M_pusch_rb, Ant_Ports)
N_sc_rb   = 12;      %% number of subcarriers in each resource block
M_pusch_sc = N_sc_rb * M_pusch_rb;



for P = 1 : Ant_Ports
	idx = 0;
	N_symbs_per_slot=7;
	for L = 0:(N_symbs_per_slot*2)-1
		if(3 == L)
			for k=0:M_pusch_sc-1
				% DMRS
				%subframe(L+1,k+1,P) = dmrs_1(P,k+1);
				subframe(k+1,L+1,P) = dmrs_1(k+1);
			end
		elseif(10 == L)
			for k=0:M_pusch_sc-1
				% DMRS
				%subframe(L+1,k+1,P) = dmrs_2(P,k+1);
				subframe(k+1,L+1,P) = dmrs_2(k+1);

			end
		else
			for k=0:M_pusch_sc-1
				% PUSCH
				subframe(k+1,L+1,P) = data(P,idx+1);
				idx                  = idx + 1;
			end
		end
	end
	
end

end
