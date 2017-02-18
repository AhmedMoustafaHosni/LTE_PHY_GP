clear; clc;
%% generate bits
rng(5); 
bits1 = randi([0,1],1,1728);
bits2 = randi([0,1],1,1728);
bits = [bits1; bits2].';

%% channel encoding
% [coded_bits1, coded_bits2]= convolutional_encoder_mimo(bits, 2);

%% Interleaver
% interleaver_input = [coded_bits1 coded_bits2];
interleaved_bits = channel_interleaver_MIMO(bits, [], [], 2, 2);
interleaved_bits_2_col = reshape(interleaved_bits,length(interleaved_bits)/2,2);
%% Scrambling

% Physical layer cell identity (we need for generation of random sequence)
N_id_cell = 2;       %% assume enodeB scheduled cell 2 for the UE
N_sc_rb   = 12;      %% number of subcarriers in each resource block
M_pusch_rb = 6;      %% number of resource blocks assigned to the UE
M_pusch_sc = M_pusch_rb*N_sc_rb;  %% total number of subcarriers
Nc = 1600;        %% specified by the 36211 v8 standard for random sequence generation      
n_s = 0;          %% assume UE send on time slot 4
n_RNTI = 10;      %% radio network temporary identifier given to the UE by enodeB (assume 10)

c_init = n_RNTI * 2^14 + floor(n_s/2) * 2^9 + N_id_cell;
c = generate_psuedo_random_seq(c_init, length(bits));
scrambled = scrambler_MIMO(interleaved_bits_2_col.', [c;  c], 2);
%% Modulation mapper
% modulated_symbols = mapper_MIMO(scrambled, 'qpsk',2);

%% layer mapper
% X = layer_mapping(modulated_symbols, 0, 0, 0, 1);

%% transform precoding
% transform_precoded_symbols = transform_precoder_mimo(X,125,2);

%% precoding
% precoded_symbols = precoding_mimo(transform_precoded_symbols,2,2);
%% SCFDMA generation



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Receiver %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SCFDMA degeneration


%% predecoding
% predecoded_symbols = predecoding_mimo(precoded_symbols,2,2);

%% transform predecoding
% transform_predecoded_symbols = transform_predecoder_mimo(predecoded_symbols,125,2);

%% layer demapper
% [layer1, layer2] = layer_demapping(transform_predecoded_symbols,2);

%% demapper
% demapped_bits = transpose(demapper_hard_MIMO(modulated_symbols, 'qpsk', 2));
%error = sum(abs(demapped_bits2 - coded_bits2));
%% descrambling
c_init = n_RNTI * 2^14 + floor(n_s/2) * 2^9 + N_id_cell;
c = generate_psuedo_random_seq(c_init, length(bits));
descrambled_bits = descrambler_MIMO(scrambled, [c; c], 2);

%% deinterleaver

[out, ri_bits, ack_bits] = channel_deinterleaver_MIMO(descrambled_bits.', 0, 0, 2, 2);
out2 = reshape(out,length(out)/2,2);
%% channel decoder
% out = transpose(viterbi_decoder_mimo(demapped_bits1, demapped_bits2, 0, 0, 2));

error = sum(abs(out2 - bits))