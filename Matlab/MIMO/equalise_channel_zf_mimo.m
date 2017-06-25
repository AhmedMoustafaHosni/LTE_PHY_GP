% Function:    equalised_symbols_zf
% Description: equalise the channel effecct on the received signal
% Inputs:      z_est                - received subframes without demodulation signal in one row
%              ce_tot               - estimated channels for each antenna
% Outputs:     equalised_symbols    

%By  : Khaled Ahmed Ali 

function equalised_symbols = equalise_channel_zf_mimo(z_est, ce_tot)
    [N_ant, M_ap_symb] = size(ce_tot);
    for m=1:N_ant
        for n=0:M_ap_symb-1
            equalised_symbols(m,n+1) = z_est(m,n+1)/ce_tot(m,n+1);
        end
    end
end