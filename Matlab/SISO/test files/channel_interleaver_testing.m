%% Testing
% Here, we generate some bits and run the interleaver code followed by the
% deinterleaver and check if data didn't change
clear; clc;
%% 1- Interleaver
N_l = 1;            % Number of Layers
Q_m = 2;            % QPSK Modulation
N_ri_bits = 12;     % #RI bit (Should be multiples of 12)
                    % as the # of columns of the constructed matrix = 12
                    
data_bits = randi([0 1],1,288*Q_m);

if (N_ri_bits == 0)
    ri_bits = [];
else
    ri_bits = randi([0 1],1,N_ri_bits*Q_m);
end

ack_bits = [];                          %No ACK bits since they replace the original data (to be able to compare
                                        %with the original data after deinterleaving)
                                        
interleaved_bits = channel_interleaver(data_bits, ri_bits, ack_bits, N_l, Q_m);

%% 2- Deinterleaver

in_bits = interleaved_bits;       %output of the interleaver
% N_ri_bits = 12;                   %12 RI bit as we did in the interleaver
N_ack_bits = 0;                   %No ACK bits were used

[deinterleaved_bits, out_ri_bits, out_ack_bits] = channel_deinterleaver(in_bits, N_ri_bits, N_ack_bits, N_l, Q_m);

%% Comparison
isequal(data_bits,deinterleaved_bits)
isequal(ri_bits,out_ri_bits)