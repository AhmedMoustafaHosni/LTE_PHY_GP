% Function:    sc_fdma_demodulator
% Description: Generates complex symbols from the sc-fdma symbols
% Inputs:      pusch_bb        - sc-fdma symbols
%              N_ul_rb      - numer of resource blocks assigned to ue
% Outputs:     symbs           - output symbols 

%edit: 25/1/2017
%By  : Ahmed Moustafa

function symbs =  sc_fdma_demodulator(pusch_bb, N_ul_rb)

FFT_size   = 2048;
N_sc_rb   = 12;      %% number of subcarriers in each resource block
N_symbs_per_slot=7;
N_cp_L_0 = 160;
N_cp_L_else = 144;
start_loc = 0;
prb_offset=0;
N_prb = N_ul_rb;

downsample = 1;
if(N_ul_rb == 6)
    FFT_pad_size = 988;
elseif(N_ul_rb == 15)
    FFT_pad_size = 934;
elseif(N_ul_rb == 25)
    FFT_pad_size = 874;
elseif(N_ul_rb == 50)
    FFT_pad_size = 724;
elseif(N_ul_rb == 75)
    FFT_pad_size = 574;
else
    FFT_pad_size = 424;
end
    
    
 for(L=0:(N_symbs_per_slot*2)-1)
     if(mod(L, 7) == 0)
         N_cp = N_cp_L_0 / downsample;
     else
         N_cp = N_cp_L_else / downsample;
     end
     index = (FFT_size + (N_cp_L_else / downsample)) * L;
     if(L > 7)
         index = index + (32 / downsample);
     elseif(L > 0)
         index = index + (16 / downsample);%1
     end
     index = index + start_loc;
     
     tmp = fftshift(fft(pusch_bb(index+N_cp+1:index+N_cp+FFT_size), FFT_size)); % extract the real symboles without CP
     tmp = tmp(1:end);%despreading
     
     tmp_symbs = tmp(FFT_pad_size+1:FFT_size-FFT_pad_size);%de- padding
     
     symbs(L+1,:) = tmp_symbs(prb_offset*12+1:prb_offset*12+(N_prb*12));% remapping to the grid again

 end


end
