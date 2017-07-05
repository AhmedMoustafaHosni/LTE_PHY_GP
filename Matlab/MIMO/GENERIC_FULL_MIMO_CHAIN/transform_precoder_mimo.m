% Function:    transform_precoder_mimo
% Description: perform transform precoding on complex data after layer mapper
% Inputs:      data         - complex data matrix with dim (M_symbs_layer * v)
%              M_pusch_rb   - numer of resource blocks assigned to ue
%			   N_layers     - number of layers
%
% Outputs:     y            - transform precodded data matrix with dim (M_symbs_layer * v)
% where M_symbs_layer is the number of symbols per layer
%            v        is the number of layers
% Edit by: Mohammed Osama 

function [y] = transform_precoder_mimo(data, M_pusch_rb, N_layers)

 N_sc_rb   = 12;      %% number of subcarriers in each resource block
 M_pusch_sc = N_sc_rb * M_pusch_rb;

 %% perform transform precoding on each layer
 for i = 1:N_layers
	 %%% transform precoder
	 % divide the symbols to number of sets depending on the number of subcarriers available to the UE for
	 % transmission.
	 % Each set represents SC_FDMA symbol.
	 % get the frequency components of each set from 0 to 2pi divided to M_PUSCH_SC sample and save them in z
	 % then get the freq. components for the next set from 0 to 2pi also
	 % the final result is a vector having M_pusch_sc samples from each set.
	 
	 for L = 0:(length(data)/M_pusch_sc)-1
		 y(L*M_pusch_sc+1:(L+1)*M_pusch_sc,i) = 1/sqrt(M_pusch_sc) * fft(data(L*M_pusch_sc+1:(L+1)*M_pusch_sc,i));
	 end
 end
end