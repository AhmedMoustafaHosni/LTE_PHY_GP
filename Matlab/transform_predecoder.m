% Function:    transform_predecoder
% Description: perform transform predecoding on complex data after sc-fdma demdulation
% Inputs:      data         - complex data output from sc-fdma demdulator
%              M_pusch_rb   - numer of resource blocks assigned to ue
% Outputs:     x            - transform predecodded data
% edit 27/1/2017
%by Ahmed Moustafa

function x = transform_predecoder(data, M_pusch_rb)

 N_sc_rb   = 12;      %% number of subcarriers in each resource block
 M_pusch_sc = N_sc_rb * M_pusch_rb;
 

    for(L=0:(length(data)/M_pusch_sc)-1)
        x(L*M_pusch_sc+1:(L+1)*M_pusch_sc) = sqrt(M_pusch_sc) * ifft(data(L*M_pusch_sc+1:(L+1)*M_pusch_sc));
    end
    
end