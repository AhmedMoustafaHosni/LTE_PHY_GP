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


% generate DMRS 


% mapping


%precoding 


% multiplexing the dmrs with data

% generate sc-fdma signal


%----------------------------reciever------------------------------------

% sc-fdma demodulation

% mapping the symbols to one vector


% generate dmrs

% demultiplex dmrs signal from the received symbols


%channel estimation and equalization



% predecoding


%demapping


%descrammpling

