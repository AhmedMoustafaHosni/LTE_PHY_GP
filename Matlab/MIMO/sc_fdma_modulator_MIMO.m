% Function:    sc_fdma_modulator
% Description: Generates sc-fdma signal of the subframe
% Inputs:      input_subframe        - received DMRS number 1 (each layer's
%                                      data in a new matrix, i.e:
%                                      input_subframe is 3D matrix)
%                                      for ex; for Layer 2's data use: input_subframe(:,:,2)
%              M_pusch_rb            - numer of resource blocks assigned to ue (each layer's data in a row)
% Outputs:     pusch_bb              - base band signal (each layer's data in a row)

%edit: 16/2/2017
%By  : Ahmed Moustafa


function pusch_bb = sc_fdma_modulator_MIMO(input_subframe , M_pusch_rb)

 N_sc_rb   = 12;      %% number of subcarriers in each resource block
 FFT_size=2048;
 prb_offset=0;
 L=0;
 N_ul_rb =M_pusch_rb;
 N_cp_L_0 = 160;
 N_cp_L_else = 144;
 N_symbs_per_slot=7;
 
     % Baseband signal generation
     for i = 1:N_layers
         last_idx = 1;
         for(L=0:(N_symbs_per_slot*2)-1)
             ifft_input_vec                 = zeros(1,FFT_size);% note that it isn't the same size
             start                          = FFT_size/2 - (N_ul_rb*N_sc_rb/2) + (prb_offset*N_sc_rb);
             stop                           = start + M_pusch_rb(i)*12;
             ifft_input_vec(start+1:stop)   = input_subframe(L+1,:,i);

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
            pusch_bb(i,last_idx:last_idx+length(cp_output)-1) = cp_output;
            % figure 
            % stem(abs(final_out));
             last_idx = last_idx + length(cp_output);
         end
     end

end
