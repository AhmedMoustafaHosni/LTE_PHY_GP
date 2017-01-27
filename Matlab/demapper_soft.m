% Function:    demapper
% Description: Maps complex-valued modulation symbols to
%              binary digits usign soft decision
% Inputs:      symbs    - Complex-valued modulation symbols
%              mod_type - Modulation type (bpsk, qpsk, 16qam,
%                         or 64qam)
% Outputs:     bits     - Binary digits
% edit 27/1/2017
%by Ahmed Moustafa

function [bits] = demapper_soft(symbs, mod_type)

    N_symbs = length(symbs);

    if( strcmp(mod_type,'bpsk') )
        % 36.211 Section 7.1.1 v10.1.0
        for(n=0:N_symbs-1)
            ang = angle(symbs(n+1));
            if(ang > -pi/4 && ang < 3*pi/4)
                act_symb  = +1/sqrt(2) + 1i/sqrt(2);
                sd        = get_soft_decision(symbs(n+1), act_symb, 1);
                bits(n+1) = +sd;
            else
                act_symb  = -1/sqrt(2) - 1i/sqrt(2);
                sd        = get_soft_decision(symbs(n+1), act_symb, 1);
                bits(n+1) = -sd;
            end
        end
    elseif( strcmp(mod_type,'qpsk') )
        % 36.211 Section 7.1.2 v10.1.0
        for(n=0:N_symbs-1)
            ang = angle(symbs(n+1));
            if(ang >= 0 && ang < pi/2)
                act_symb      = +1/sqrt(2) + 1i/sqrt(2);
                sd            = get_soft_decision(symbs(n+1), act_symb, 1);
                bits(n*2+0+1) = +sd;
                bits(n*2+1+1) = +sd;
            elseif(ang >= -pi/2 && ang < 0)
                act_symb      = +1/sqrt(2) - 1i/sqrt(2);
                sd            = get_soft_decision(symbs(n+1), act_symb, 1);
                bits(n*2+0+1) = +sd;
                bits(n*2+1+1) = -sd;
            elseif(ang >= pi/2 && ang < pi)
                act_symb      = -1/sqrt(2) + 1i/sqrt(2);
                sd            = get_soft_decision(symbs(n+1), act_symb, 1);
                bits(n*2+0+1) = -sd;
                bits(n*2+1+1) = +sd;
            else
                act_symb      = -1/sqrt(2) - 1i/sqrt(2);
                sd            = get_soft_decision(symbs(n+1), act_symb, 1);
                bits(n*2+0+1) = -sd;
                bits(n*2+1+1) = -sd;
            end
        end
    elseif( strcmp(mod_type,'16qam') )
        % 36.211 Section 7.1.3 v10.1.0
        %ERROR: Not supporting 16qam at this time
        bits = 0;
    elseif( strcmp(mod_type,'64qam') )
        % 36.211 Section 7.1.4 v10.1.0
        %ERROR: Not supporting 64qam at this time
        bits = 0;
    end
end

function [sd] = get_soft_decision(p1, p2, max_dist)
    dist = sqrt((real(p1)-real(p2))^2 + (imag(p1)-imag(p2))^2);

    if(dist >= max_dist)
        dist = max_dist - (max_dist/1000);
    end

    sd = max_dist - dist;
end
