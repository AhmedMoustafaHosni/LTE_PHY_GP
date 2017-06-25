clear; clc;
M_PUSCH_SC = 12;
RBs = 6;
NumAnt = 2;
%% generate bits
rng(5); 
bits1 = randi([0,1],1,1728);
bits2 = randi([0,1],1,1728);
bits = [bits1 bits2];

%% channel encoding
% [coded_bits1, coded_bits2]= convolutional_encoder_mimo(bits, 2);

% Interleaver
% interleaver_input = [coded_bits1 coded_bits2];
interleaved_bits = channel_interleaver_MIMO([bits1; bits2].', [], [], 2, 2);
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
c = generate_psuedo_random_seq(c_init, length(interleaved_bits_2_col));
scrambled = scrambler_MIMO(interleaved_bits_2_col.', [c;  c], 2);
%% DMRs gebneration
dmrs_a = generate_dmrs_pusch(0, 2, 0, 0, 0, 0, 0, 'fixed', M_pusch_rb, 0);
dmrs_1_a = dmrs_a(1:M_pusch_sc);
dmrs_2_a = dmrs_a(M_pusch_sc+1:2*M_pusch_sc);

dmrs_b = generate_dmrs_pusch(0, 2, 0, 0, 0, 0, 0, 'fixed', M_pusch_rb, 1); % diversity layer inquiry
dmrs_1_b = dmrs_b(1:M_pusch_sc);
dmrs_2_b = dmrs_b(M_pusch_sc+1:2*M_pusch_sc);
%% Modulation mapper
% input : 2 rows
% output : 2 rows
modulated_symbols1 = mapper(scrambled(1,:), 'qpsk');
modulated_symbols2 = mapper(scrambled(2,:), 'qpsk');

%% layer mapper
X = layer_mapping(modulated_symbols1, modulated_symbols2, 0, 0, 2);
% 
%% transform precoding
transform_precoded_symbols = transform_precoder_mimo(X,6,2);
% 
%% precoding
precoded_symbols = precoding_mimo(transform_precoded_symbols,2,2);
%% generate a matrix of size 2(antennas) * Number of subcarriers assigned to the UE
% dmrs1 is used to fill symbol 4. one row for each grid (for each antenna)
dmrs1 = [dmrs_1_a; dmrs_1_b];
% dmrs2 is used to fill symbol 11. one row for each grid (for each antenna)
dmrs2 = [dmrs_2_a; dmrs_2_b];

%% compose the grid for each antenna
subframe_per_ant = compose_subframe_mimo(precoded_symbols,dmrs1,dmrs2,RBs,2);

%% SCFDMA generation
tx_signal = sc_fdma_modulator_MIMO(subframe_per_ant, RBs, 2);
%__________________________________________________________________________________%
%% Channel model
chcfg.DelayProfile = 'EPA';
chcfg.NRxAnts = 2;
chcfg.DopplerFreq = 5;
chcfg.MIMOCorrelation = 'Low';
chcfg.Seed = 1;
chcfg.InitPhase = 'Random';
chcfg.ModelType = 'GMEDS';
chcfg.NTerms = 16;
chcfg.NormalizeTxAnts = 'On';
chcfg.NormalizePathGains = 'On';
chcfg.SamplingRate = 30.72 * 10^6;
chcfg.InitTime = 0;
txSignalAfterChannel = lteFadingChannel(chcfg,tx_signal.').';
% txSignalAfterChannel = tx_signal;
SNR = [-20:0];
estimated_noise_power = zeros(length(SNR),2);
for i = 1 : length(SNR)
	N0 = 0.5 / (10^(SNR(i)/10));
    N = sqrt(N0/4)/ sqrt(double(2048)); %* sqrt(6*12);

   % Add noise
    noise1 = N*complex(randn(size(txSignalAfterChannel(1,:))), randn(size(txSignalAfterChannel(1,:))));
    noise2 = N*complex(randn(size(txSignalAfterChannel(1,:))), randn(size(txSignalAfterChannel(1,:))));
    noise = [noise1 ; noise2];
    txSignalWithNoise = txSignalAfterChannel+ noise;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Receiver %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SCFDMA degeneration
demod_signal = sc_fdma_demodulator_MIMO(txSignalWithNoise, RBs, 2);
%
%% DMRS 
% extract dmrs from rx'ed signals
DMRS_1_RxAnten_1 = demod_signal(:,1+3,1).';
DMRS_2_RxAnten_1 = demod_signal(:,1+10,1).';
DMRS_1_RxAnten_2 = demod_signal(:,1+3,2).';
DMRS_2_RxAnten_2 = demod_signal(:,1+10,2).';
DMR_rx_1 = [DMRS_1_RxAnten_1 ; DMRS_1_RxAnten_2];
DMR_rx_2 = [DMRS_2_RxAnten_1 ; DMRS_2_RxAnten_2];
%% channel estimation and equalization
[ channel,estimated_noise_power(i,:)] = estimate_channel_mimo(DMR_rx_1, DMR_rx_2, dmrs1, dmrs2, M_pusch_sc, 7, NumAnt);
%% Decompose subframe
Decoded_streams = decompose_subframe_mimo(demod_signal, 2);
%% channel equalization 
equalised_subframe = equalise_channel_zf_mimo(Decoded_streams, channel);
%% predecoding
predecoded_symbols = predecoding_mimo(equalised_subframe,2,2); 
%% transform predecoding
transform_predecoded_symbols = transform_predecoder_mimo(predecoded_symbols,6,2);
%% layer demapper
[layer1, layer2] = layer_demapping(transform_predecoded_symbols,2);

%% demapper
demapped_bits1 = transpose(demapper_hard(layer1, 'qpsk'));
demapped_bits2 = transpose(demapper_hard(layer2, 'qpsk'));
%error = sum(abs(demapped_bits2 - coded_bits2));
%% descrambling
descrambled_bits = descrambler_MIMO([demapped_bits1 demapped_bits2].', [c; c], 2);


%% deinterleaver
[out, ri_bits, ack_bits] = channel_deinterleaver_MIMO(descrambled_bits.', 0, 0, 2, 2);
out2 = reshape(out,length(out)/2,2);

%% channel decoder
% out = transpose(viterbi_decoder_mimo(out2(:,1), out2(:,2), 0, 0, 2));
BER(i) = sum(abs(out - bits))/length(bits);
%error = sum(abs(out - bits))
theoBER(i) = berawgn(SNR(i),'psk',4,'nondiff');
% 	
end
%% *Curves*

figure;
set(gca,'fontsize', 14);
h=semilogy(SNR,BER,'-red',SNR,theoBER,'-blue');
set(h,'linewidth', 2);
legend(h,'BER_Q_P_S_K','Theoretical BER_Q_P_S_K');
ylabel('BER'); 
xlabel('Eb/No');
title('Theoretical BER vs Actual BER');

% 
% error = sum(bits-out)