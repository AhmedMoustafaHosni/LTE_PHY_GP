% Function:    transform_predecoder_mimo
% Description: perform transform predecoding on complex data after predecoding
% Inputs:      data         - complex data output from predecoder
%              M_pusch_rb   - numer of resource blocks assigned to ue
%			   N_layers     - number of layers
%
% Outputs:     x            - transform predecodded data matrix with dim (M_symbs_layer * v)
% where M_symbs_layer is the number of symbols per layer
%            v        is the number of layers
% Edit by: Mohammed Osama 


function x = transform_predecoder_mimo(data, M_pusch_rb, N_layers)

 N_sc_rb   = 12;      %% number of subcarriers in each resource block
 M_pusch_sc = N_sc_rb * M_pusch_rb;
 
%% perform transform predecoding on the predecoded matrix
 for i = 1:N_layers
    for L=0:(length(data)/M_pusch_sc)-1
        x(L*M_pusch_sc+1:(L+1)*M_pusch_sc,i) = sqrt(M_pusch_sc) * ifft(data(L*M_pusch_sc+1:(L+1)*M_pusch_sc,i));
    end
 end
end