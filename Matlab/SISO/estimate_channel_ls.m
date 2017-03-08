% Function:    estimate_channel
% Description: Generates channel estimation 
% Inputs:      symb_0                - received DMRS number 1
%              symb_1                - received DMRS number 2
%              dmrs_0                - generated DMRS number 1
%              dmrs_1                - generated DMRS number 2
%              M_pusch_sc            - number of subcarriers allocated to ue
%              N_symbs_per_slot      - number of symbols per slot
% Outputs:     channel - channel estimation matrix to be used for equalization

%edit: 25/1/2017
%By  : Ahmed Moustafa

function [ channel noise ] = estimate_channel_ls(symb_0, symb_1, dmrs_0, dmrs_1, M_pusch_sc, N_symbs_per_slot)

    % Determine channel estimates
    for m=0:M_pusch_sc-1 
        H1(m+1)        = symb_0(m+1)/dmrs_0(m+1);
        mag_0(m+1) = abs( H1(m+1) );
        ang_0(m+1) = angle( H1(m+1) );

        H2(m+1)         = symb_1(m+1)/dmrs_1(m+1);
        mag_1(m+1) = abs( H2(m+1) );
        ang_1(m+1) = angle( H2(m+1) );
    end
    
    H_time_averaged = (H1 + H2)/2;
    
    for ii = 0:M_pusch_sc-1 
        if (ii == 0 || ii == (M_pusch_sc-1) )
            
            H_freq_averaged(ii+1) = H_time_averaged(ii+1);
        
        elseif (ii == 1 || ii == (M_pusch_sc-1-1) )
            
            H_freq_averaged(ii+1) = sum(H_time_averaged((ii+1-1):(ii+1+1)))/3 ;
            
        elseif (ii == 2 || ii == (M_pusch_sc-1-2) )
            
            H_freq_averaged(ii+1) = sum(H_time_averaged((ii+1-2):(ii+1+2)))/5 ;
            
        elseif (ii == 3 || ii == (M_pusch_sc-1-3) )
            
            H_freq_averaged(ii+1) = sum(H_time_averaged((ii+1-3):(ii+1+3)))/7 ;
            
        elseif (ii == 4 || ii == (M_pusch_sc-1-4) )
            
            H_freq_averaged(ii+1) = sum(H_time_averaged((ii+1-4):(ii+1+4)))/9 ;
            
        elseif (ii == 5 || ii == (M_pusch_sc-1-5) )
            
            H_freq_averaged(ii+1) = sum(H_time_averaged((ii+1-5):(ii+1+5)))/11 ;
            
        elseif (ii == 6 || ii == (M_pusch_sc-1-6) )
            
            H_freq_averaged(ii+1) = sum(H_time_averaged((ii+1-6):(ii+1+6)))/13 ;
            
        elseif (ii == 7 || ii == (M_pusch_sc-1-7) )
            
            H_freq_averaged(ii+1) = sum(H_time_averaged((ii+1-7):(ii+1+7)))/15 ;
            
        elseif (ii == 8 || ii == (M_pusch_sc-1-8) )
            
            H_freq_averaged(ii+1) = sum(H_time_averaged((ii+1-8):(ii+1+8)))/17 ;
            
        else
                
            H_freq_averaged(ii+1) = sum(H_time_averaged((ii+1-9):(ii+1+9)))/19 ;
            
        end
    end 

    
    
    noise_estimate = H_time_averaged - H_freq_averaged ;
    
    noise = mean(noise_estimate.*conj(noise_estimate));
    
    channel = [ H_freq_averaged(1,:),  H_freq_averaged(1,:),  H_freq_averaged(1,:),  H_freq_averaged(1,:),  H_freq_averaged(1,:),  H_freq_averaged(1,:),  H_freq_averaged(1,:),  H_freq_averaged(1,:),  H_freq_averaged(1,:),  H_freq_averaged(1,:),  H_freq_averaged(1,:),  H_freq_averaged(1,:) ];
end