% This file contain 5 different tests, run each one alone
%% Test 1
% Here, we generate some bits and run the interleaver code followed by the
% deinterleaver and check if data didn't change

clear; clc;

% 1- Interleaver
N_l = 2;            % Number of Layers
Q_m = 6;           % Modulation Order (2=QPSK, 4=16QAM, 6=64QAM)
N_ri_bits = 12;     % #RI bit (Should be multiples of 12)
                    % as the # of columns of the constructed matrix = 12
                    
data_bits = randi([0 1],1,288*Q_m*N_l);

if (N_ri_bits == 0)
    ri_bits = [];
else
    ri_bits = randi([0 1],1,N_ri_bits*Q_m*N_l);
end

ack_bits = [];                          %No ACK bits since they replace the original data (to be able to compare
                                        %with the original data after deinterleaving)
                                        
interleaved_bits = channel_interleaver(data_bits, ri_bits, ack_bits, N_l, Q_m);

% 2- Deinterleaver
in_bits = interleaved_bits;       %output of the interleaver
% N_ri_bits = 12;                   %12 RI bit as we did in the interleaver
N_ack_bits = 0;                   %No ACK bits were used

[deinterleaved_bits, out_ri_bits, out_ack_bits] = channel_deinterleaver(in_bits, N_ri_bits, N_ack_bits, N_l, Q_m);

% Comparison
isequal(data_bits,deinterleaved_bits)
isequal(ri_bits,out_ri_bits)
%% Test 2
% Here, we generate some bits and run the interleaver code followed by the
% deinterleaver and check if data didn't change.
% But in this test, we assign the ri_bits as a sequence of known numbers
% to check if RI bits are put in their correct place as the standard states
% Thus, (interleaved_array) variable is to be checked.
clear; clc;
% 1- Interleaver
N_l = 2;            % Number of Layers
Q_m = 6;           % Modulation Order (2=QPSK, 4=16QAM, 6=64QAM)
N_ri_bits = 12;     % #RI bit (Should be multiples of 12)
                    % as the # of columns of the constructed matrix = 12

data_bits = randi([0 0],1,288*Q_m*N_l);
ri_bits = 1:1:(N_ri_bits*Q_m*N_l);
ack_bits = [];                          %No ACK bits since they replace the original data (to be able to compare
                                        %with the original data after deinterleaving)
                                        
interleaved_bits = channel_interleaver(data_bits, ri_bits, ack_bits, N_l, Q_m);

% 2- Deinterleaver

in_bits = interleaved_bits;       %output of the interleaver
N_ack_bits = 0;                   %No ACK bits were used

[deinterleaved_bits, out_ri_bits, out_ack_bits] = channel_deinterleaver(in_bits, N_ri_bits, N_ack_bits, N_l, Q_m);

% Comparison
isequal(data_bits,deinterleaved_bits)
isequal(ri_bits,out_ri_bits)
interleaved_array = reshape(interleaved_bits,length(interleaved_bits)/12,12);

%% Test 3
% Testing our code against the MATLAB Built-in Code
clear; clc;

N_l = 2;            % Number of Layers
Q_m = 6;           % Modulation Order (2=QPSK, 4=16QAM, 6=64QAM)
N_ri_bits = 12;     % #RI bit (Should be multiples of 12)
                    % as the # of columns of the constructed matrix = 12
N_ack_bits = 0;     % No ACK bits
                      
data_bits = randi([0 1],1,288*Q_m*N_l);

if (N_ri_bits == 0)
    ri_bits = [];
else
    ri_bits = randi([0 1],1,N_ri_bits*Q_m*N_l);
end

ack_bits = [];                          %No ACK bits since they replace the original data (to be able to compare
                                        %with the original data after deinterleaving)
% 1.1- Interleaver (Our Code)

interleaved = channel_interleaver(data_bits, ri_bits, ack_bits, N_l, Q_m);
% 1.2- Interleaver (MATLAB)

ue.CyclicPrefixUL = 'Normal';
ue.Shortened = 0;

chs.Modulation = '64QAM';       %(16QAM, 64QAM, QPSK)
chs.NLayers = N_l;
% chs.QdCQI = 0;
chs.QdRI = N_ri_bits;
% chs.QdACK = 0;

cqi = [];

interleaved_MATLAB = lteULSCHInterleave(ue, chs, data_bits, cqi, ri_bits, ack_bits).';
% 2.1- Deinterleaver (Our Code)

[deinterleaved, out_ri_bits, out_ack_bits] = channel_deinterleaver(interleaved, N_ri_bits, N_ack_bits, N_l, Q_m);

% 2.2- Deinterleaver (MATLAB)

[deinterleaved_MATLAB,out_cqi_MATLAB,out_ri_MATLAB,out_ack_MATLAB] = lteULSCHDeinterleave(ue,chs,interleaved_MATLAB);

% Comparison
isequal(interleaved,interleaved_MATLAB)
isequal(deinterleaved,deinterleaved_MATLAB)

%% Test 4
% Testing our code against the MATLAB Built-in Code by comparing the
% constructed matrix in each case with all data bits = 0, and RI bits are a
% sequence of known numbers.
clear; clc;

N_l = 2;            % Number of Layers
Q_m = 6;           % Modulation Order (2=QPSK, 4=16QAM, 6=64QAM)
N_ri_bits = 12;     % #RI bit (Should be multiples of 12)
                    % as the # of columns of the constructed matrix = 12
N_ack_bits = 0;     % No ACK bits
                      
data_bits = randi([0 0],1,288*Q_m*N_l);

ri_bits = 1:1:(N_ri_bits*Q_m*N_l);

ack_bits = [];                          %No ACK bits since they replace the original data (to be able to compare
                                        %with the original data after deinterleaving)
% 1.1- Interleaver (Our Code)

interleaved = channel_interleaver(data_bits, ri_bits, ack_bits, N_l, Q_m);
% 1.2- Interleaver (MATLAB)

ue.CyclicPrefixUL = 'Normal';
ue.Shortened = 0;

chs.Modulation = '64QAM';
chs.NLayers = N_l;
% chs.QdCQI = 0;
chs.QdRI = N_ri_bits;
% chs.QdACK = 0;

cqi = [];

interleaved_MATLAB = lteULSCHInterleave(ue, chs, data_bits, cqi, ri_bits, ack_bits).';
% 2.1- Deinterleaver (Our Code)

[deinterleaved, out_ri_bits, out_ack_bits] = channel_deinterleaver(interleaved, N_ri_bits, N_ack_bits, N_l, Q_m);

% 2.2- Deinterleaver (MATLAB)

[deinterleaved_MATLAB,out_cqi_MATLAB,out_ri_MATLAB,out_ack_MATLAB] = lteULSCHDeinterleave(ue,chs,interleaved_MATLAB);

% Comparison
disp('Comparing interleaved bits');
isequal(interleaved,interleaved_MATLAB)
disp('Comparing deinterleaved bits');
isequal(deinterleaved,deinterleaved_MATLAB)

% To check if the RI bits are inserted at the right places:

mat = reshape(interleaved,length(interleaved)/12,12);
mat_MATLAB = reshape(interleaved_MATLAB,length(interleaved_MATLAB)/12,12);

disp('See: mat & mat_MATLAB');

%% Test 5
% We test interleaving then deinterleaving data bits & RI bits using only
% MATLAB Built-in Functions.

% PUSCH Interleave
% Interleave two PUSCH RBs worth of bits for QPSK modulation. Considering
% the REs reserved for PUSCH DM-RS, there are 144 REs available for PUSCH
% data per RB.
% 144 REs: (12 subcarriers)*(12 symbols out of 14 total symobls for PUSCH)
% Therefore, two RBs contain 288 PUSCH symbols.  This results
% in 2*288 bits to QPSK modulate after interleaving.

%
% Initialize UE specific and UL-SCH related parameter structures. Generate
% data for QPSK modulation of PUSCH symbols in two RBs. For QPSK, there are
% two bits per symbol. Perform interleaving and symbol modulation.
% 
clc;
clear;

ue.CyclicPrefixUL = 'Normal';
ue.Shortened = 0;
% Shorten subframe, specified as 0 or 1. If 1, the last symbol of the
% subframe is not used. It should be set if the current subframe contains a
% possible SRS transmission.
%
chs.Modulation = '64QAM';       %(16QAM, 64QAM, QPSK)
chs.NLayers = 2;
chs.QdCQI = 0;        % # of QI bits
chs.QdRI = 12;        % # of RI bits (Should be multiples of 12)
                      % as the # of columns of the constructed matrix = 12
chs.QdACK = 0;        % # of ACK bits
%
numRB = 1;
numREperRB = 144;
bitsPerSymbol = 2;
numBits = numRB * numREperRB * bitsPerSymbol;

input_data = randi([0 1], numBits * bitsPerSymbol, 1); 

%No CQI,ACK in the moment

N_l = 2;           % Number of Layers
Q_m = 6;           % Modulation Order (2=QPSK, 4=16QAM, 6=64QAM)

cqi = [];
ri = 1:1:(chs.QdRI*Q_m*N_l);
ack = []; 
%
% interleaved = lteULSCHInterleave(ue, chs, input_data);
interleaved = lteULSCHInterleave(ue, chs, input_data, cqi, ri, ack);
% interleaved = interleaved';     %To compare it with our written code

% symbols_tx = lteSymbolModulate(interleaved, 'QPSK');
% Deinterleaving
% symbols_rx = lteSymbolDemodulate(symbols_tx,'QPSK');
[out_data,out_cqi,out_ri,out_ack] = lteULSCHDeinterleave(ue,chs,interleaved);

% Comparison
disp('Comparing input & output data bits');
isequal(input_data,out_data)
disp('Comparing input & output RI bits');
isequal(ri.',out_ri)
