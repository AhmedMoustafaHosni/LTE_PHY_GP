% Function:    estimate_channel_mimo
% Description: Generates channel estimation 
% Inputs:      symb_0                - received DMRS number 1  for each port
%              symb_1                - received DMRS number 2  for each port
%              dmrs_0                - generated DMRS number 1 for each port
%              dmrs_1                - generated DMRS number 2 for each port
%              M_pusch_sc            - number of subcarriers allocated to ue
%              N_symbs_per_slot      - number of symbols per slot
%              N_rxAnt               - number of rx'ed antenna
% Outputs:     channel               - channel estimation tensor to be used for equalization
%              noise                 - estimated noise power
%By  : Khaled Ahmed Ali 

function [ channel noise ] = estimate_channel_mimo(symb_0, symb_1, dmrs_0, dmrs_1, M_pusch_sc, N_symbs_per_slot, N_rxAnt)

    % Determine channel estimates
    for n=1:N_rxAnt
        for m=0:M_pusch_sc-1
            H1(n,m+1)    = symb_0(n,m+1)/dmrs_0(n,m+1);
            mag_0(n,m+1) = abs( H1(n,m+1) );
            ang_0(n,m+1) = angle( H1(n,m+1) );
            
            H2(n,m+1)    = symb_1(n,m+1)/dmrs_1(n,m+1);
            mag_1(n,m+1) = abs( H2(n,m+1) );
            ang_1(n,m+1) = angle( H2(n,m+1) );
        end
        
        H_time_averaged(n,:) = (H1(n,:) + H2(n,:))/2;
        
        for ii = 0:M_pusch_sc-1
            if (ii == 0 || ii == (M_pusch_sc-1) )
                
                H_freq_averaged(n,ii+1) = H_time_averaged(n,ii+1);
                
            elseif (ii == 1 || ii == (M_pusch_sc-1-1) )
                
                H_freq_averaged(n,ii+1) = sum(H_time_averaged(n,(ii+1-1):(ii+1+1)))/3 ;
                
            elseif (ii == 2 || ii == (M_pusch_sc-1-2) )
                
                H_freq_averaged(n,ii+1) = sum(H_time_averaged(n,(ii+1-2):(ii+1+2)))/5 ;
                
            elseif (ii == 3 || ii == (M_pusch_sc-1-3) )
                
                H_freq_averaged(n,ii+1) = sum(H_time_averaged(n,(ii+1-3):(ii+1+3)))/7 ;
                
            elseif (ii == 4 || ii == (M_pusch_sc-1-4) )
                
                H_freq_averaged(n,ii+1) = sum(H_time_averaged(n,(ii+1-4):(ii+1+4)))/9 ;
                
            elseif (ii == 5 || ii == (M_pusch_sc-1-5) )
                
                H_freq_averaged(n,ii+1) = sum(H_time_averaged(n,(ii+1-5):(ii+1+5)))/11 ;
                
            elseif (ii == 6 || ii == (M_pusch_sc-1-6) )
                
                H_freq_averaged(n,ii+1) = sum(H_time_averaged(n,(ii+1-6):(ii+1+6)))/13 ;
                
            elseif (ii == 7 || ii == (M_pusch_sc-1-7) )
                
                H_freq_averaged(n,ii+1) = sum(H_time_averaged(n,(ii+1-7):(ii+1+7)))/15 ;
                
            elseif (ii == 8 || ii == (M_pusch_sc-1-8) )
                
                H_freq_averaged(n,ii+1) = sum(H_time_averaged(n,(ii+1-8):(ii+1+8)))/17 ;
                
            else
                
                H_freq_averaged(n,ii+1) = sum(H_time_averaged(n,(ii+1-9):(ii+1+9)))/19 ;
                
            end
        end
        
        
        
        noise_estimate(n,:) = H_time_averaged(n,:) - H_freq_averaged(n,:) ;
        
        noise(n) = mean(noise_estimate(n,:).*conj(noise_estimate(n,:)));
        
%         noise = noise.';
        
        channel(n,:) = [ H_freq_averaged(n,:),  H_freq_averaged(n,:),  H_freq_averaged(n,:),  H_freq_averaged(n,:),  H_freq_averaged(n,:),  H_freq_averaged(n,:),  H_freq_averaged(n,:),  H_freq_averaged(n,:),  H_freq_averaged(n,:),  H_freq_averaged(n,:),  H_freq_averaged(n,:),  H_freq_averaged(n,:) ];
    end
end