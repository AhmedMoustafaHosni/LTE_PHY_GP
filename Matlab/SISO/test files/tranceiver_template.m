clc;
clear all;
close all;

% size of bits
M_bits = 1728;  %% just to send one subframe see page 16 in 36211 release 8
% generate random bits of size M_bits
rng(5);
b = randi([0 1],1,M_bits);

% Physical layer cell identity (we need for generation of random sequence)
N_id_cell = 2;       %% assume enodeB scheduled cell 2 for the UE
N_sc_rb   = 12;      %% number of subcarriers in each resource block
M_pusch_rb = 6;      %% number of resource blocks assigned to the UE
M_pusch_sc = M_pusch_rb*N_sc_rb;  %% total number of subcarriers
Nc = 1600;        %% specified by the 36211 v8 standard for random sequence generation      
n_s = 0;          %% assume UE send on time slot 4
n_RNTI = 10;      %% radio network temporary identifier given to the UE by enodeB (assume 10)


%-------------------------transmitter-----------------------------------

% scrambling
c_init = n_RNTI * 2^14 + floor(n_s/2) * 2^9 + N_id_cell;
c = generate_psuedo_random_seq(c_init, M_bits);
b_scrampled = scrambler(b, c);

% generate DMRS
dmrs = generate_dmrs_pusch(0, N_id_cell, 0, 0, 0, 0, 0, 'fixed', M_pusch_rb, 0);
dmrs_1 = dmrs(1:M_pusch_sc);
dmrs_2 = dmrs(M_pusch_sc+1:2*M_pusch_sc);

% mapping
mapped_data = mapper( b_scrampled , 'qpsk' );

%precoding 
precoded_data = transform_precoder(mapped_data, M_pusch_rb);

% multiplexing the dmrs with data
subframe_1 = compose_subframe(precoded_data, dmrs_1, dmrs_2, M_pusch_rb);

% generate sc-fdma signal
modulated_subframe = sc_fdma_modulator(subframe_1, M_pusch_rb);


%-----------------------channel and noise--------------------------------







%----------------------------reciever------------------------------------

% sc-fdma demodulation
demodulated_subframe = sc_fdma_demodulator(modulated_subframe, M_pusch_rb);

% mapping the symbols to one vector
demodulated_subframe_vect =[demodulated_subframe(0+1,:), demodulated_subframe(1+1,:), demodulated_subframe(2+1,:), demodulated_subframe(4+1,:), demodulated_subframe(5+1,:), demodulated_subframe(6+1,:), demodulated_subframe(7+1,:), demodulated_subframe(8+1,:), demodulated_subframe(9+1,:), demodulated_subframe(11+1,:), demodulated_subframe(12+1,:), demodulated_subframe(13+1,:)];


% generate dmrs
dmrs = generate_dmrs_pusch(0, 2, 0, 0, 0, 0, 0, 'fixed', 6, 0);
dmrs_1 = dmrs(1:M_pusch_sc);
dmrs_2 = dmrs(M_pusch_sc+1:2*M_pusch_sc);

% demultiplex dmrs signal from the received symbols
dmrs_1_rx = demodulated_subframe(1+3,:);
dmrs_2_rx = demodulated_subframe(1+10,:);

%channel estimation and equalization
channel = estimate_channel(dmrs_1_rx, dmrs_2_rx, dmrs_1, dmrs_2, M_pusch_sc, 7);
equalised_subframe = equalise_channel(demodulated_subframe_vect, channel);

% predecoding
predecoded_data = transform_predecoder(equalised_subframe, M_pusch_rb);

%demapping
demapped_data = demapper_hard(predecoded_data, 'qpsk' );

%descrammpling
c_init = n_RNTI * 2^14 + floor(n_s/2) * 2^9 + N_id_cell;
c = generate_psuedo_random_seq(c_init, M_bits);
received_bits = descrambler(demapped_data, c);

isequal(b,received_bits)

