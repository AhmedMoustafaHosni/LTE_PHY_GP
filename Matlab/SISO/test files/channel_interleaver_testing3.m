%% Testing
% Testing our code against the MATLAB Built-in Code by comparing the output
% of both the inter. and deinter.
clear; clc;

N_l = 1;            % Number of Layers
Q_m = 2;            % QPSK Modulation
N_ri_bits = 12;     % #RI bit (Should be multiples of 12)
                    % as the # of columns of the constructed matrix = 12
N_ack_bits = 0;     % No ACK bits
                      
data_bits = randi([0 1],1,288*Q_m);

if (N_ri_bits == 0)
    ri_bits = [];
else
    ri_bits = randi([0 1],1,N_ri_bits*Q_m);
end

ack_bits = [];                          %No ACK bits since they replace the original data (to be able to compare
                                        %with the original data after deinterleaving)
%% 1.1- Interleaver (Our Code)

interleaved = channel_interleaver(data_bits, ri_bits, ack_bits, N_l, Q_m);
%% 1.2- Interleaver (MATLAB)

ue.CyclicPrefixUL = 'Normal';
ue.Shortened = 0;

chs.Modulation = 'QPSK';
chs.NLayers = N_l;
% chs.QdCQI = 0;
chs.QdRI = N_ri_bits;
% chs.QdACK = 0;

cqi = [];

interleaved_MATLAB = lteULSCHInterleave(ue, chs, data_bits, cqi, ri_bits, ack_bits).';
%% 2.1- Deinterleaver (Our Code)

[deinterleaved, out_ri_bits, out_ack_bits] = channel_deinterleaver(interleaved, N_ri_bits, N_ack_bits, N_l, Q_m);

%% 2.2- Deinterleaver (MATLAB)

[deinterleaved_MATLAB,out_cqi_MATLAB,out_ri_MATLAB,out_ack_MATLAB] = lteULSCHDeinterleave(ue,chs,interleaved_MATLAB);

%% Comparison
isequal(interleaved,interleaved_MATLAB)
isequal(deinterleaved,deinterleaved_MATLAB)