% NOIE: THE OLD "precoding_mimo" BLOCK in MIMO folder in github IS MODIFIED

clc;
clear;

N_l = 64; 
Q_m = 6; 
mod_type = '64qam'; 
N_sc_rb   = 12;      % number of subcarriers in each resource block
M_pusch_rb = 100;      % number of resource blocks assigned to the UE
M_pusch_sc = M_pusch_rb*N_sc_rb;  % total number of subcarriers
N_bits = N_l * M_pusch_sc * N_sc_rb * Q_m;

% %to get the input data from file
% data_bits_total = (fread(fopen('1.txt')) - '0').';
% data_bits = reshape(data_bits_total,length(data_bits_total)/N_l,N_l);
% ri_bits = (fread(fopen('ri_0.txt'))-'0').';

rng(5);
data_bits_total = randi([0 1],1,N_bits);
data_bits = reshape(data_bits_total,length(data_bits_total)/N_l,N_l);
ri_bits=[];

% Interleaver
interleaved_bits = channel_interleaver_MIMO(data_bits, ri_bits, [], N_l, Q_m);
interleaved_bits_Nlayer_col = reshape(interleaved_bits,length(interleaved_bits)/N_l,N_l);
%generate_psuedo_random_seq
c_init = 10 * 2 ^ 14 + floor(0 / 2) * 2 ^ 9 + 2; 
c = generate_psuedo_random_seq(c_init, length(interleaved_bits_Nlayer_col)); 
% Scrambling
scrambled = scrambler_MIMO(interleaved_bits_Nlayer_col.', repmat(c,N_l,1), N_l); 
% Modulation mapper
for i = 1:N_l
modulated_symbols(:,i) = mapper(scrambled(i,:), mod_type).';
end
% transform precoding
transform_precoded_symbols = transform_precoder_mimo(modulated_symbols, M_pusch_rb, N_l);
% precoding
precoded_symbols = precoding_mimo(transform_precoded_symbols,N_l,N_l);
% generate a matrix of size 2(antennas) * Number of subcarriers assigned to the UE
for i = 1:N_l
    dmrs(i,:) = generate_dmrs_pusch(0, 2, 0, 0, 0, 0, 0, 'fixed', M_pusch_rb, mod((i-1),4));
end
dmrs1 = dmrs(:,1:M_pusch_sc);
dmrs2 = dmrs(:,M_pusch_sc+1:2*M_pusch_sc);
% compose the grid for each antenna
subframe_per_ant = compose_subframe_mimo(precoded_symbols, dmrs1, dmrs2, M_pusch_rb, N_l);
% SCFDMA generation
symbols_MATLAB = sc_fdma_modulator_MIMO(subframe_per_ant, M_pusch_rb, N_l);

%for testing in cuda
%symbols_MATLAB = reshape(symbols_MATLAB,1,length(symbols_MATLAB)*N_l);
%sum((abs(symbols_MATLAB) - abs(symbols_CUDA)))
