% Function:    equalised_symbols_MMSE
% Description: equalise the channel effecct on the received signal
% Inputs:      z_est                - received subframes without demodulation signal in one row
%              ce_tot               - estimated channels
%              N0                   - noise power
% Outputs:     equalised symbols

%By  : Khaled Ahmed Ali 

function equalised_symbols = equalise_channel_MMSE_mimo(z_est, ce_tot, N0)
    
    [N_ant, M_ap_symb] = size(ce_tot);
    for m=1:N_ant
        noise_matrix = N0(1,m)*eye(M_ap_symb);
        T = ce_tot(m,:)' * ce_tot(m,:);
        G = inv(T+noise_matrix) * ce_tot(m,:)';
        G = G.';
        
        for n=0:M_ap_symb-1
            equalised_symbols(m,n+1) = z_est(m,n+1)*G(n+1);
        end
    end
end