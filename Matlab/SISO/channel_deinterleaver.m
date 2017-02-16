%
% Copyright 2014 Ben Wojtowicz
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU Affero General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU Affero General Public License for more details.
%
%    You should have received a copy of the GNU Affero General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
% Function:    lte_ulsch_channel_deinterleaver
% Description: Deinterleaves ULSCH data from RI and ACK control information
% Inputs:      in_bits    - Interleaved bits
%              N_ri_bits  - Number of RI control bits to deinterleave
%              N_ack_bits - Number of ACK control bits to deinterleave
%              N_l        - Number of layers
%              Q_m        - Number of bits per modulation symbol
% Outputs:     data_bits - Deinterleaved data bits
%              ri_bits   - Deinterleaved RI control bits
%              ack_bits  - Deinterleaved ACK control bits
% Spec:        3GPP TS 36.212 section 5.2.2.8 v10.1.0
% Notes:       Only supporting normal CP
% Rev History: Ben Wojtowicz 03/26/2014 Created
%
%% For Testing Purpose

% N_l = 1;            %Number of Layers
% Q_m = 2;            %QPSK Modulation
% 
% in_bits = out_bits;       %output of the interleaver
% N_ri_bits = 12;           %12 RI bit as we did in the interleaver
% N_ack_bits = 0;           %No ACK bits were used

%%
function [data_bits, ri_bits, ack_bits] = channel_deinterleaver(in_bits, N_ri_bits, N_ack_bits, N_l, Q_m)

    N_pusch_symbs  = 12;
    ri_column_set  = [1, 4, 7, 10];
    ack_column_set = [2, 3, 8, 9];

    % Step 1: Define C_mux
    C_mux = N_pusch_symbs;

    % Step 2: Define R_mux and R_prime_mux
    % We've replaced lines (51&52) with lines (53&54) as they gave errors
%     H_prime       = length(in_bits)/(Q_m*N_l);        %we edited this line
%     H_prime_total = H_prime + N_ri_bits;              %we edited this line
    H_prime_total = length(in_bits)/(Q_m*N_l);          %we added this line instead
    H_prime = H_prime_total - N_ri_bits;                %we added this line instead
    R_mux         = (H_prime_total*Q_m*N_l)/C_mux;
    R_prime_mux   = R_mux/(Q_m*N_l);
    
    % Initialize the matricies
    y_idx = -10000*ones(1,C_mux*R_prime_mux);
    y_mat = zeros(1,C_mux*R_mux);

    % Step 6: Construct matrix
    idx = 0;
    for(n=0:C_mux-1)
        for(m=0:R_prime_mux-1)
            for(k=0:(Q_m*N_l)-1)
                y_mat(m*C_mux*Q_m*N_l + n*Q_m*N_l + k + 1) = in_bits(idx+1);
                idx                                        = idx + 1;
            end
        end
    end

    % Step 5: Deinterleave the ACK control bits
    n = 0;
    m = 0;
    r = R_prime_mux-1;
    if(N_ack_bits == 0)
        ack_bits = 0;
    end
    while(n < N_ack_bits)
        C_ack = ack_column_set(m+1);
        for(k=0:(Q_m*N_l)-1)
            ack_bits(n+1,k+1) = y_mat(C_mux*r*Q_m*N_l + C_ack*Q_m*N_l + k + 1);
        end
        y_idx(C_mux*r + C_ack + 1) = 2;
        n                          = n + 1;
        r                          = R_prime_mux - 1 - floor(n/4);
        m                          = mod(m+3, 4);
    end

    % Step 3: Deinterleave the RI control bits
    n = 0;
    m = 0;
    r = R_prime_mux-1;
    if(N_ri_bits == 0)
        ri_bits = [];
    end
    while(n < N_ri_bits)
        C_ri = ri_column_set(m+1);
        for(k=0:(Q_m*N_l)-1)
            ri_bits(n+1,k+1) = y_mat(C_mux*r*Q_m*N_l + C_ri*Q_m*N_l + k + 1);
        end
        y_idx(C_mux*r + C_ri + 1) = 1;
        n                         = n + 1;
        r                         = R_prime_mux - 1 - floor(n/4);
        m                         = mod(m+3, 4);
        end

    % Step 4: Interleave the data bits
    n = 0;
    k = 0;
    while(k < H_prime)
        if(y_idx(n+1) == -10000)
            y_idx(n+1) = 1;
            for(m=0:(Q_m*N_l)-1)
                data_bits(k+1,m+1) = y_mat(n*(Q_m*N_l) + m + 1);
            end
            k = k + 1;
            elseif(y_idx(n+1) == 2)
                data_bits(k+1,:) = -10000*ones(1,Q_m*N_l);
                k                = k + 1;
            end
        n = n + 1;
            end
            
            data_bits = reshape(data_bits.',length(data_bits)*Q_m,1).';
            ri_bits = reshape(ri_bits.',length(ri_bits)*Q_m,1).';
end
