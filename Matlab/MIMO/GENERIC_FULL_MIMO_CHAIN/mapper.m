% Function:    modulation_mapper
% Description: Maps binary digits to complex-valued modulation symbols
% Inputs:      bits     - Binary digits to map
%              mod_type - Modulation type (bpsk, qpsk, 16qam, or 64qam)
% Outputs:     symbs    - Complex-valued modulation symbols
% edit 27/1/2017
%
function [symbs] = mapper(bits, mod_type)

    N_bits = length(bits);

    if( strcmp(mod_type,'bpsk') )
        % 36.211 Section 7.1.1 v10.1.0
        for(n=0:N_bits-1)
            input = bits(n+1);
            if(input == 0)
                symbs(n+1) = +1/sqrt(2) + 1i/sqrt(2);
            else % input == 1
                symbs(n+1) = -1/sqrt(2) - 1i/sqrt(2);
            end
        end
    elseif( strcmp(mod_type,'qpsk') )
        % 36.211 Section 7.1.2 v10.1.0
        if(mod(N_bits, 2) == 0)
            loc_bits = bits;
        else
            loc_bits = [bits, zeros(1, 2-mod(N_bits, 2))];
            N_bits   = N_bits + 2-mod(N_bits, 2);
        end
        for(n=0:(N_bits/2)-1)
            input = loc_bits(n*2+1)*2 + loc_bits(n*2+1+1);
            if(input == 0)
                symbs(n+1) = +1/sqrt(2) + 1i/sqrt(2);
            elseif(input == 1)
                symbs(n+1) = +1/sqrt(2) - 1i/sqrt(2);
            elseif(input == 2)
                symbs(n+1) = -1/sqrt(2) + 1i/sqrt(2);
            else % input == 3
                symbs(n+1) = -1/sqrt(2) - 1i/sqrt(2);
            end
        end
    elseif( strcmp(mod_type,'16qam') )
        % 36.211 Section 7.1.3 v10.1.0
        if(mod(N_bits, 4) == 0)
            loc_bits = bits;
        else
            loc_bits = [bits, zeros(1, 4-mod(N_bits, 4))];
            N_bits   = N_bits + 4-mod(N_bits, 4);
        end
        for(n=0:(N_bits/4)-1)
            input = loc_bits(n*4+1)*8 + loc_bits(n*4+1+1)*4 + loc_bits(n*4+2+1)*2 + loc_bits(n*4+3+1);
            if(input == 0)
                symbs(n+1) = +1/sqrt(10) + 1i*1/sqrt(10);
            elseif(input == 1)
                symbs(n+1) = +1/sqrt(10) + 1i*3/sqrt(10);
            elseif(input == 2)
                symbs(n+1) = +3/sqrt(10) + 1i*1/sqrt(10);
            elseif(input == 3)
                symbs(n+1) = +3/sqrt(10) + 1i*3/sqrt(10);
            elseif(input == 4)
                symbs(n+1) = +1/sqrt(10) - 1i*1/sqrt(10);
            elseif(input == 5)
                symbs(n+1) = +1/sqrt(10) - 1i*3/sqrt(10);
            elseif(input == 6)
                symbs(n+1) = +3/sqrt(10) - 1i*1/sqrt(10);
            elseif(input == 7)
                symbs(n+1) = +3/sqrt(10) - 1i*3/sqrt(10);
            elseif(input == 8)
                symbs(n+1) = -1/sqrt(10) + 1i*1/sqrt(10);
            elseif(input == 9)
                symbs(n+1) = -1/sqrt(10) + 1i*3/sqrt(10);
            elseif(input == 10)
                symbs(n+1) = -3/sqrt(10) + 1i*1/sqrt(10);
            elseif(input == 11)
                symbs(n+1) = -3/sqrt(10) + 1i*3/sqrt(10);
            elseif(input == 12)
                symbs(n+1) = -1/sqrt(10) - 1i*1/sqrt(10);
            elseif(input == 13)
                symbs(n+1) = -1/sqrt(10) - 1i*3/sqrt(10);
            elseif(input == 14)
                symbs(n+1) = -3/sqrt(10) - 1i*1/sqrt(10);
            else % input == 15
                symbs(n+1) = -3/sqrt(10) - 1i*3/sqrt(10);
            end
        end
    elseif( strcmp(mod_type,'64qam') )
        % 36.211 Section 7.1.4 v10.1.0
        if(mod(N_bits, 6) == 0)
            loc_bits = bits;
        else
            loc_bits = [bits, zeros(1, 6-mod(N_bits, 6))];
            N_bits   = N_bits + 6-mod(N_bits, 6);
        end
        for(n=0:(N_bits/6)-1)
            input = loc_bits(n*6+1)*32 + loc_bits(n*6+1+1)*16 + loc_bits(n*6+2+1)*8 + loc_bits(n*6+3+1)*4 + loc_bits(n*6+4+1)*2 + loc_bits(n*6+5+1);
            if(input == 0)
                symbs(n+1) = +3/sqrt(42) + 1i*3/sqrt(42);
            elseif(input == 1)
                symbs(n+1) = +3/sqrt(42) + 1i*1/sqrt(42);
            elseif(input == 2)
                symbs(n+1) = +1/sqrt(42) + 1i*3/sqrt(42);
            elseif(input == 3)
                symbs(n+1) = +1/sqrt(42) + 1i*1/sqrt(42);
            elseif(input == 4)
                symbs(n+1) = +3/sqrt(42) + 1i*5/sqrt(42);
            elseif(input == 5)
                symbs(n+1) = +3/sqrt(42) + 1i*7/sqrt(42);
            elseif(input == 6)
                symbs(n+1) = +1/sqrt(42) + 1i*5/sqrt(42);
            elseif(input == 7)
                symbs(n+1) = +1/sqrt(42) + 1i*7/sqrt(42);
            elseif(input == 8)
                symbs(n+1) = +5/sqrt(42) + 1i*3/sqrt(42);
            elseif(input == 9)
                symbs(n+1) = +5/sqrt(42) + 1i*1/sqrt(42);
            elseif(input == 10)
                symbs(n+1) = +7/sqrt(42) + 1i*3/sqrt(42);
            elseif(input == 11)
                symbs(n+1) = +7/sqrt(42) + 1i*1/sqrt(42);
            elseif(input == 12)
                symbs(n+1) = +5/sqrt(42) + 1i*5/sqrt(42);
            elseif(input == 13)
                symbs(n+1) = +5/sqrt(42) + 1i*7/sqrt(42);
            elseif(input == 14)
                symbs(n+1) = +7/sqrt(42) + 1i*5/sqrt(42);
            elseif(input == 15)
                symbs(n+1) = +7/sqrt(42) + 1i*7/sqrt(42);
            elseif(input == 16)
                symbs(n+1) = +3/sqrt(42) - 1i*3/sqrt(42);
            elseif(input == 17)
                symbs(n+1) = +3/sqrt(42) - 1i*1/sqrt(42);
            elseif(input == 18)
                symbs(n+1) = +1/sqrt(42) - 1i*3/sqrt(42);
            elseif(input == 19)
                symbs(n+1) = +1/sqrt(42) - 1i*1/sqrt(42);
            elseif(input == 20)
                symbs(n+1) = +3/sqrt(42) - 1i*5/sqrt(42);
            elseif(input == 21)
                symbs(n+1) = +3/sqrt(42) - 1i*7/sqrt(42);
            elseif(input == 22)
                symbs(n+1) = +1/sqrt(42) - 1i*5/sqrt(42);
            elseif(input == 23)
                symbs(n+1) = +1/sqrt(42) - 1i*7/sqrt(42);
            elseif(input == 24)
                symbs(n+1) = +5/sqrt(42) - 1i*3/sqrt(42);
            elseif(input == 25)
                symbs(n+1) = +5/sqrt(42) - 1i*1/sqrt(42);
            elseif(input == 26)
                symbs(n+1) = +7/sqrt(42) - 1i*3/sqrt(42);
            elseif(input == 27)
                symbs(n+1) = +7/sqrt(42) - 1i*1/sqrt(42);
            elseif(input == 28)
                symbs(n+1) = +5/sqrt(42) - 1i*5/sqrt(42);
            elseif(input == 29)
                symbs(n+1) = +5/sqrt(42) - 1i*7/sqrt(42);
            elseif(input == 30)
                symbs(n+1) = +7/sqrt(42) - 1i*5/sqrt(42);
            elseif(input == 31)
                symbs(n+1) = +7/sqrt(42) - 1i*7/sqrt(42);
            elseif(input == 32)
                symbs(n+1) = -3/sqrt(42) + 1i*3/sqrt(42);
            elseif(input == 33)
                symbs(n+1) = -3/sqrt(42) + 1i*1/sqrt(42);
            elseif(input == 34)
                symbs(n+1) = -1/sqrt(42) + 1i*3/sqrt(42);
            elseif(input == 35)
                symbs(n+1) = -1/sqrt(42) + 1i*1/sqrt(42);
            elseif(input == 36)
                symbs(n+1) = -3/sqrt(42) + 1i*5/sqrt(42);
            elseif(input == 37)
                symbs(n+1) = -3/sqrt(42) + 1i*7/sqrt(42);
            elseif(input == 38)
                symbs(n+1) = -1/sqrt(42) + 1i*5/sqrt(42);
            elseif(input == 39)
                symbs(n+1) = -1/sqrt(42) + 1i*7/sqrt(42);
            elseif(input == 40)
                symbs(n+1) = -5/sqrt(42) + 1i*3/sqrt(42);
            elseif(input == 41)
                symbs(n+1) = -5/sqrt(42) + 1i*1/sqrt(42);
            elseif(input == 42)
                symbs(n+1) = -7/sqrt(42) + 1i*3/sqrt(42);
            elseif(input == 43)
                symbs(n+1) = -7/sqrt(42) + 1i*1/sqrt(42);
            elseif(input == 44)
                symbs(n+1) = -5/sqrt(42) + 1i*5/sqrt(42);
            elseif(input == 45)
                symbs(n+1) = -5/sqrt(42) + 1i*7/sqrt(42);
            elseif(input == 46)
                symbs(n+1) = -7/sqrt(42) + 1i*5/sqrt(42);
            elseif(input == 47)
                symbs(n+1) = -7/sqrt(42) + 1i*7/sqrt(42);
            elseif(input == 48)
                symbs(n+1) = -3/sqrt(42) - 1i*3/sqrt(42);
            elseif(input == 49)
                symbs(n+1) = -3/sqrt(42) - 1i*1/sqrt(42);
            elseif(input == 50)
                symbs(n+1) = -1/sqrt(42) - 1i*3/sqrt(42);
            elseif(input == 51)
                symbs(n+1) = -1/sqrt(42) - 1i*1/sqrt(42);
            elseif(input == 52)
                symbs(n+1) = -3/sqrt(42) - 1i*5/sqrt(42);
            elseif(input == 53)
                symbs(n+1) = -3/sqrt(42) - 1i*7/sqrt(42);
            elseif(input == 54)
                symbs(n+1) = -1/sqrt(42) - 1i*5/sqrt(42);
            elseif(input == 55)
                symbs(n+1) = -1/sqrt(42) - 1i*7/sqrt(42);
            elseif(input == 56)
                symbs(n+1) = -5/sqrt(42) - 1i*3/sqrt(42);
            elseif(input == 57)
                symbs(n+1) = -5/sqrt(42) - 1i*1/sqrt(42);
            elseif(input == 58)
                symbs(n+1) = -7/sqrt(42) - 1i*3/sqrt(42);
            elseif(input == 59)
                symbs(n+1) = -7/sqrt(42) - 1i*1/sqrt(42);
            elseif(input == 60)
                symbs(n+1) = -5/sqrt(42) - 1i*5/sqrt(42);
            elseif(input == 61)
                symbs(n+1) = -5/sqrt(42) - 1i*7/sqrt(42);
            elseif(input == 62)
                symbs(n+1) = -7/sqrt(42) - 1i*5/sqrt(42);
            else % input == 63
                symbs(n+1) = -7/sqrt(42) - 1i*7/sqrt(42);
            end
        end
    end
end
