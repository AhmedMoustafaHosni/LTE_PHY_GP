%% PUSCH Interleave
% Interleave two PUSCH RBs worth of bits for QPSK modulation. Considering
% the REs reserved for PUSCH DM-RS, there are 144 REs available for PUSCH
% data per RB.
% 144 REs: (12 subcarriers)*(12 symbols out of 14 total symobls for PUSCH)
% Therefore, two RBs contain 288 PUSCH symbols.  This results
% in 2*288 bits to QPSK modulate after interleaving.
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
% Function:    lte_ulsch_channel_interleaver
% Description: Interleaves ULSCH data with RI and ACK control information
% Inputs:      data_bits - Data bits to interleave
%              ri_bits   - RI control bits to interleave
%              ack_bits  - ACK control bits to interleave
%              N_l       - Number of layers
%              Q_m       - Number of bits per modulation symbol
% Outputs:     out_bits - Interleaved bits
% Spec:        3GPP TS 36.212 section 5.2.2.8 v10.1.0
% Notes:       Only supporting normal CP
% Rev History: Ben Wojtowicz 03/26/2014 Created
%
%% For Testing Purpose

% N_l = 2;            %Number of Layers
% Q_m = 2;            %QPSK Modulation
% 
% data_bits = randi([0 1],288,Q_m);
% ri_bits = randi([0 1],12,Q_m);          %Testing with 12 RI bit
% ack_bits = [];                          %No ACK bits since they replace the original data (to be able to compare
%                                         %with the original data after deinterleaving)
%%
function [out_bits] = channel_interleaver_MIMO(ip_data_bits, ip_ri_bits, ack_bits, N_l, Q_m)
        
    data_bits = [];
    ri_bits = [];
        
    for i = 1:N_l
        data_bits_cons(:,:,i) = reshape(ip_data_bits(:,i),Q_m,size(ip_data_bits(:,i),1)/(Q_m)).';
        data_bits = [data_bits data_bits_cons(:,:,i)];

        if(~isempty(ip_ri_bits))
            ri_bits_cons(:,:,i) = reshape(ip_ri_bits(:,i),Q_m,size(ip_ri_bits(:,i),1)/(Q_m)).';
            ri_bits = [ri_bits ri_bits_cons(:,:,i)];
        else
            ri_bits = [];
        end
        
    end
  
    N_pusch_symbs  = 12;
    ri_column_set  = [1, 4, 7, 10];
    ack_column_set = [2, 3, 8, 9];

    [N_data_bits, data_vec_len] = size(data_bits);
    [N_ri_bits, ri_vec_len]     = size(ri_bits);
    [N_ack_bits, ack_vec_len]   = size(ack_bits);

    % Step 1: Define C_mux
    C_mux = N_pusch_symbs;

    % Step 2: Define R_mux and R_prime_mux
    H_prime   = N_data_bits;
    H_vec_len = data_vec_len;
    if(ri_vec_len == H_vec_len)
        H_prime_total = H_prime + N_ri_bits;
    else
        H_prime_total = H_prime;
    end
    R_mux       = (H_prime_total*Q_m*N_l)/C_mux;
    R_prime_mux = R_mux/(Q_m*N_l);

    % Initialize the matricies
    y_idx = -10000*ones(1,C_mux*R_prime_mux);
    y_mat = zeros(1,C_mux*R_mux);

    % Step 3: Interleave the RI control bits
    if(ri_vec_len == H_vec_len)
        n = 0;
        m = 0;
        r = R_prime_mux-1;
        while(n < N_ri_bits)
            C_ri                  = ri_column_set(m+1);
            y_idx(r*C_mux+C_ri+1) = 1;
            for(k=0:(Q_m*N_l)-1)
                y_mat(C_mux*r*Q_m*N_l + C_ri*Q_m*N_l + k + 1) = ri_bits(n+1,k+1);
            end
            n = n + 1;
            r = R_prime_mux - 1 - floor(n/4);
            m = mod(m+3, 4);
        end
    end

    % Step 4: Interleave the data bits
    n = 0;
    k = 0;
    while(k < H_prime)
        if(y_idx(n+1) == -10000)
            y_idx(n+1) = 1;
            for(m=0:(Q_m*N_l)-1)
                y_mat(n*(Q_m*N_l) + m + 1) = data_bits(k+1,m+1);
            end
            k = k + 1;
        end
        n = n + 1;
    end

    % Step 5: Interleave the ACK control bits
    if(ack_vec_len == H_vec_len)
        n = 0;
        m = 0;
        r = R_prime_mux-1;
        while(n < N_ack_bits)
            C_ack                  = ack_column_set(m+1);
            y_idx(r*C_mux+C_ack+1) = 2;
            for(k=0:(Q_m*N_l)-1)
                y_mat(C_mux*r*Q_m*N_l + C_ack*Q_m*N_l + k + 1) = ack_bits(n+1,k+1);
            end
            n = n + 1;
            r = R_prime_mux - 1 - floor(n/4);
            m = mod(m+3, 4);
        end
    end

    % Step 6: Read out the bits
    idx = 0;
    for(n=0:C_mux-1)
        for(m=0:R_prime_mux-1)
            for(k=0:H_vec_len-1)
                out_bits(idx+1) = y_mat(m*C_mux*Q_m*N_l + n*Q_m*N_l + k + 1);
                idx             = idx + 1;
            end
        end
    end
    
    %For MIMO part
    out_bits = reshape(out_bits,length(out_bits)/N_l,N_l).';
    
end
