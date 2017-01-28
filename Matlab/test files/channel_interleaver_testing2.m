%% Testing
% Here, we generate some bits and run the interleaver code followed by the
% deinterleaver and check if data didn't change
clear; clc;
%% 1- Interleaver
N_l = 1;            %Number of Layers
Q_m = 2;            %QPSK Modulation

data_bits = randi([0 0],1,288*Q_m);
% ri_bits = randi([0 1],12,Q_m);          %Testing with 12 RI bit
ri_bits = [1 2;3 4;5 6; 7 8; 9 10; 11 12; 13 14; 15 16; 17 18; 19 20; 21 22; 23 24];
ack_bits = [];                          %No ACK bits since they replace the original data (to be able to compare
                                        %with the original data after deinterleaving)
                                        
interleaved_bits = channel_interleaver(data_bits, ri_bits, ack_bits, N_l, Q_m);

%% 2- Deinterleaver
N_l = 1;            %Number of Layers
Q_m = 2;            %QPSK Modulation

in_bits = interleaved_bits;       %output of the interleaver
N_ri_bits = 12;                   %12 RI bit as we did in the interleaver
N_ack_bits = 0;                   %No ACK bits were used

[deinterleaved_bits, out_ri_bits, out_ack_bits] = channel_deinterleaver(in_bits, N_ri_bits, N_ack_bits, N_l, Q_m);

%% Comparison
isequal(data_bits,deinterleaved_bits)
isequal(ri_bits,out_ri_bits)
reshape(interleaved_bits,length(interleaved_bits)/12,12);