clear
%% generate bits
rng(5); 
bits1 = randi([0,1],1,1000);
bits2 = randi([0,1],1,1000);
bits = [bits1 bits2];

%% channel encoding
[coded_bits1, coded_bits2]= convolutional_encoder_mimo(bits, 2);

%% Interleaver


%% Scrambling


%% Modulation mapper
modulated_symbols1 = mapper(coded_bits1, 'qpsk');
modulated_symbols2 = mapper(coded_bits2, 'qpsk');

%% layer mapper
X = layer_mapping(modulated_symbols1, modulated_symbols2, 0, 0, 2);

%% transform precoding
transform_precoded_symbols = transform_precoder_mimo(X,125,2);

%% precoding
precoded_symbols = precoding_mimo(transform_precoded_symbols,2,2);
%% SCFDMA generation



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Receiver %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SCFDMA degeneration


%% predecoding
predecoded_symbols = predecoding_mimo(precoded_symbols,2,2);

%% transform predecoding
transform_predecoded_symbols = transform_predecoder_mimo(predecoded_symbols,125,2);

%% layer demapper
[layer1, layer2] = layer_demapping(transform_predecoded_symbols,2);

%% demapper
demapped_bits1 = transpose(demapper_hard(layer1, 'qpsk'));
demapped_bits2 = transpose(demapper_hard(layer2, 'qpsk'));
%error = sum(abs(demapped_bits2 - coded_bits2));
%% descrambling


%% deinterleaver


%% channel decoder
out = transpose(viterbi_decoder_mimo(demapped_bits1, demapped_bits2, 0, 0, 2));

error = sum(abs(out - bits))