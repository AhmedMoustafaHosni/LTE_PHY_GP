clear; clc;
M_PUSCH_SC = 12;
RBs = 6;
%% generate bits
rng(5); 
bits1 = randi([0,1],1,1728/3);
bits2 = randi([0,1],1,1728/3);
bits = [bits1 bits2];

%% channel encoding
[coded_bits1, coded_bits2]= convolutional_encoder_mimo(bits, 2);

% Interleaver
interleaver_input = [coded_bits1 coded_bits2];
interleaved_bits = channel_interleaver_MIMO(interleaver_input, [], [], 2, 2);
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
%% Modulation mapper
% input : 2 rows
% output : 2 rows
modulated_symbols1 = mapper(scrambled(1,:), 'qpsk');
modulated_symbols2 = mapper(scrambled(2,:), 'qpsk');

% %% layer mapper
X = layer_mapping(modulated_symbols1, modulated_symbols2, 0, 0, 2);
% 
% %% transform precoding
transform_precoded_symbols = transform_precoder_mimo(X,6,2);
% 
% %% precoding
precoded_symbols = precoding_mimo(transform_precoded_symbols,2,2);
% 
subframe_per_ant = compose_subframe_mimo(precoded_symbols,zeros(1,RBs*M_PUSCH_SC),zeros(1,RBs*M_PUSCH_SC),RBs,2);
% %% SCFDMA generation
tx_signal = sc_fdma_modulator_MIMO(subframe_per_ant, RBs, 2);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Channel %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %tx = fft(tx_signal(1,:));
% 
SNR = [-20:0];
for i = 1 : length(SNR)
	N0 = 0.5 / (10^(SNR(i)/10));
    N = sqrt(N0/4)/ sqrt(double(2048)); %* sqrt(6*12);
    % Add noise
% %    noise = N*complex(randn(size(tx_signal)), randn(size(tx_signal)));
% % 	SNR2 = 10^(SNR(i)/20);
% %     N = 1/(SNR2*sqrt(2*2*double(2048)));
% %     % Add noise
     noise1 = N*complex(randn(size(tx_signal(1,:))), randn(size(tx_signal(1,:))));
     noise2 = N*complex(randn(size(tx_signal(1,:))), randn(size(tx_signal(1,:))));
    noise = [noise1 ; noise2];
    txSignalWithNoise = tx_signal + noise;
%    %Noise = (1/length(tx_signal(1,:)) .* ifft(noise.')).';
%    %txSignalWithNoise = tx_signal + Noise;
% 
% % tmp1 = 	awgn(tx_signal(1,:),SNR(i), 'measured');
% % tmp2 = 	awgn(tx_signal(2,:),SNR(i), 'measured');
% % txSignalWithNoise = [tmp1; tmp2];
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Receiver %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %% SCFDMA degeneration
demod_signal = sc_fdma_demodulator_MIMO(txSignalWithNoise, RBs, 2);
% 
% %% Decompose subframe
Decoded_streams = decompose_subframe_mimo(demod_signal, 2);
% %% predecoding
predecoded_symbols = predecoding_mimo(Decoded_streams,2,2);
% 
% %% transform predecoding
transform_predecoded_symbols = transform_predecoder_mimo(predecoded_symbols,6,2);
% 
% %% layer demapper
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
out = transpose(viterbi_decoder_mimo(out2(:,1), out2(:,2), 0, 0, 2));
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