% Function:    transform_precoder
% Description: perform transform precoding on complex data after mapper
% Inputs:      data         - complex data
%              M_pusch_rb   - numer of resource blocks assigned to ue
% Outputs:     z            - transform precodded data
% edit 27/1/2017
%by Ahmed Moustafa

function z = transform_precoder(data, M_pusch_rb)

 N_sc_rb   = 12;      %% number of subcarriers in each resource block
 M_pusch_sc = N_sc_rb * M_pusch_rb;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% the symbols should be mapped to different layers to support MIMO 
%%% for now we assume single antenna configuration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% transform precoder
% divide the symbols to number of sets depending on the number of subcarriers available to the UE for 
% transmission.
% Each set represents SC_FDMA symbol.
% get the frequency components of each set from 0 to 2pi divided to M_PUSCH_SC sample and save them in z
% then get the freq. components for the next set from 0 to 2pi also
% the final result is a vector having M_pusch_sc samples from each set.


    for L = 0:(length(data)/M_pusch_sc)-1
        z(L*M_pusch_sc+1:(L+1)*M_pusch_sc) = 1/sqrt(M_pusch_sc) * fft(data(L*M_pusch_sc+1:(L+1)*M_pusch_sc));
    end
    
end