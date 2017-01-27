% Function:    generate_base_band_signal
% Description: Generates sc-fdma signal of the subframe
% Inputs:      input_subframe        - received DMRS number 1
%              M_pusch_rb            - numer of resource blocks assigned to ue
% Outputs:     pusch_bb              - base band signal

%edit: 25/1/2017
%By  : Ahmed Moustafa


function pusch_bb = generate_base_band_signal(input_subframe , M_pusch_rb)

 N_sc_rb   = 12;      %% number of subcarriers in each resource block
 FFT_size=2048;
 prb_offset=0;
 L=0;
 N_ul_rb =100;
 N_cp_L_0 = 160;
 N_cp_L_else = 144;
 N_symbs_per_slot=7;
 
 % Baseband signal generation
 last_idx = 1;
 for(L=0:(N_symbs_per_slot*2)-1)
     ifft_input_vec                 = zeros(1,FFT_size);% note that it isn't the same size
     start                          = FFT_size/2 - (N_ul_rb*N_sc_rb/2) + (prb_offset*N_sc_rb*2);
     stop                           = start + M_pusch_rb*12;
     ifft_input_vec(start:stop-1)   = input_subframe(L+1,:);
     
     ifft_output_vec        = ifft(ifftshift(ifft_input_vec), FFT_size);
     resamp_ifft_output_vec = ifft_output_vec(1:end);
     % Add cyclic prefix
     if(L == 0 || L == 7)
         cp_output = [resamp_ifft_output_vec(length(resamp_ifft_output_vec)-N_cp_L_0+1:length(resamp_ifft_output_vec)), resamp_ifft_output_vec];
     else
         cp_output = [resamp_ifft_output_vec(length(resamp_ifft_output_vec)-N_cp_L_else+1:length(resamp_ifft_output_vec)), resamp_ifft_output_vec];
     end
    % figure 
    %  stem(abs(cp_output));
    % Concatenate output
    pusch_bb(last_idx:last_idx+length(cp_output)-1) = cp_output;
    % figure 
    % stem(abs(final_out));
     last_idx = last_idx + length(cp_output);
 end
 

end