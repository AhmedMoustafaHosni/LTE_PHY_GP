% Function:    demapper
% Description: Maps complex-valued modulation symbols to
%              binary digits using hard decision
% Inputs:      symbs    - Complex-valued modulation symbols (each layer's data in a row)
%              mod_type - Modulation type (bpsk, qpsk, 16qam,
%                         or 64qam)
% Outputs:     bits     - Binary digits (each layer's data in a row)
% edit 27/1/2017
%by Ahmed Moustafa

function [bits] = demapper_hard_MIMO(symbs, mod_type,N_layers)
    for i = 1:N_layers

        N_symbs = length(symbs(i,:));

        if( strcmp(mod_type,'bpsk') )
            % 36.211 Section 7.1.1 v10.1.0
            for(n=0:N_symbs-1)
                ang = angle(symbs(i,n+1));
                if(ang > -pi/4 && ang < 3*pi/4)
                    act_symb  = +1/sqrt(2) + 1i/sqrt(2);
                    bits(i,n+1) = 1;
                else
                    act_symb  = -1/sqrt(2) - 1i/sqrt(2);
                    bits(i,n+1) = 0;
                end
            end
        elseif( strcmp(mod_type,'qpsk') )
            % 36.211 Section 7.1.2 v10.1.0
            for(n=0:N_symbs-1)
                ang = angle(symbs(i,n+1));
                if(ang >= 0 && ang < pi/2)
                    act_symb      = +1/sqrt(2) + 1i/sqrt(2);
                    bits(n*2+0+1) = 0;
                    bits(n*2+1+1) = 0;
                elseif(ang >= -pi/2 && ang < 0)
                    act_symb      = +1/sqrt(2) - 1i/sqrt(2);
                    bits(n*2+0+1) = 0;
                    bits(n*2+1+1) = 1;
                elseif(ang >= pi/2 && ang < pi)
                    act_symb      = -1/sqrt(2) + 1i/sqrt(2);
                    bits(n*2+0+1) = 1;
                    bits(n*2+1+1) = 0;
                else
                    act_symb      = -1/sqrt(2) - 1i/sqrt(2);
                    bits(n*2+0+1) = 1;
                    bits(n*2+1+1) = 1;
                end
            end
        elseif( strcmp(mod_type,'16qam') )
            % 36.211 Section 7.1.3 v10.1.0
            %ERROR: Not supporting 16qam at this time
            for(n=0:N_symbs-1)
                symb_real = real(symbs(i,n+1));
                symb_imag = imag(symbs(i,n+1));

                if(symb_real < 0)
                    bits(n*4+0+1) = 1;
                else
                    bits(n*4+0+1) = 0;
                end

                if(symb_imag < 0)
                    bits(n*4+1+1) = 1;
                else
                    bits(n*4+1+1) = 0;
                end

                if(abs(symb_real) < 2/sqrt(10))
                    bits(n*4+2+1) = 0;
                else
                    bits(n*4+2+1) = 1;
                end

                if(abs(symb_imag) < 2/sqrt(10))
                    bits(n*4+3+1) = 0;
                else
                    bits(n*4+3+1) = 1;
                end
            end
        elseif( strcmp(mod_type,'64qam') )
            % 36.211 Section 7.1.4 v10.1.0
            %ERROR: Not supporting 64qam at this time
            for(n=0:N_symbs-1)
                symb_real = real(symbs(i,n+1));
                symb_imag = imag(symbs(i,n+1));

                if(symb_real < 0)
                    bits(n*6+0+1) = 1;
                else
                    bits(n*6+0+1) = 0;
                end

                if(symb_imag < 0)
                    bits(n*6+1+1) = 1;
                else
                    bits(n*6+1+1) = 0;
                end

                if(abs(symb_real) < 4/sqrt(42))
                    bits(n*6+2+1) = 0;
                else
                    bits(n*6+2+1) = 1;
                end

                if(abs(symb_imag) < 4/sqrt(42))
                    bits(n*6+3+1) = 0;
                else
                    bits(n*6+3+1) = 1;
                end

                if(abs(symb_real) > 2/sqrt(42) && abs(symb_real) < 6/sqrt(42) )
                    bits(n*6+4+1) = 0;
                else
                    bits(n*6+4+1) = 1;
                end

                if(abs(symb_imag) > 2/sqrt(42) && abs(symb_imag) < 6/sqrt(42) )
                    bits(n*6+5+1) = 0;
                else
                    bits(n*6+5+1) = 1;
                end
            end
        end
    end
end

