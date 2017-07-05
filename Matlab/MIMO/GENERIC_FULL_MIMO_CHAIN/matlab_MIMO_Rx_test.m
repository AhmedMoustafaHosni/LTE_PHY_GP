% clc;
% clear;

N_l = 64; 
Q_m = 6; 
mod_type = '64qam'; 
N_sc_rb   = 12;      % number of subcarriers in each resource block
M_pusch_rb = 100;      % number of resource blocks assigned to the UE
M_pusch_sc = M_pusch_rb*N_sc_rb;  % total number of subcarriers
N_bits = N_l * M_pusch_sc * N_sc_rb * Q_m;

%%to get the input data from file
% data_bits_total = (fread(fopen('1.txt')) - '0').';
% data_bits = reshape(data_bits_total,length(data_bits_total)/N_l,N_l);
% ri_bits = (fread(fopen('ri_0.txt'))-'0').';

%for test (i.e Run "matlab_MIMO_Rx_test"  after "matlab_MIMO_Tx_test")
txSignal = symbols_MATLAB;

% SCFDMA degeneration
demod_signal = sc_fdma_demodulator_MIMO(txSignal, M_pusch_rb, N_l);
% Decompose subframe
Decoded_streams = decompose_subframe_mimo(demod_signal, N_l);
% predecoding
predecoded_symbols = predecoding_mimo(Decoded_streams,N_l,N_l);
% transform predecoding
transform_predecoded_symbols = transform_predecoder_mimo(predecoded_symbols,M_pusch_rb,N_l);
% demapper
for i = 1:N_l
demapped_bits(i,:) = demapper_hard(transform_predecoded_symbols(:,i), mod_type);
end
% descrambling
descrambled_bits = descrambler_MIMO(demapped_bits, repmat(c,N_l,1), N_l);
% deinterleaver
[received_bits, ri_bits, ack_bits] = channel_deinterleaver_MIMO(descrambled_bits.', 0, 0, N_l, Q_m);

%for test (i.e Run "matlab_MIMO_Rx_test"  after "matlab_MIMO_Tx_test")
isequal(data_bits_total,received_bits)
