%% *Procedures of Modeling LTE Uplink Channel Encoding BER using Hard decision*
% *0. Channel Encoding* 
%
% *1. Scrambling*
% 
% *2. Modulation Mapper* (for choosing soft or hard decisions, go to the demapper part)
% 
% *3. Transform Precoder*
% 
% *4. RE Mapper*
% 
% *5. SC-FDMA Signal Generator*
%% *Constructing the UE Structure and generating random bits*

clear all;
close all;
clc;
% Create a UE-specific configuration structure, get PUSCH indices, and
% generate a bit stream sized according to configuration structure.

z_test = 0;
z_test2 = 0;

ue = lteRMCUL('A3-2');      %See Table 8.2.3.1-1 in TS 36.104
% Now, ue has: 6 RBs - Normal CP - 10 subframes - QPSK (PUSCH)
a_NULRB = 100;
a_NULRB_UE = 8;     %RBs for this specific UE
a_NCellID = 2;
a_RNTI = 10;
a_TotSubframes = 10;
a_NFrame = 0;        %Frame index
a_NSubframe = 0;

ue.NULRB = a_NULRB;
ue.NCellID = a_NCellID;
ue.RNTI = a_RNTI;
ue.TotSubframes = a_TotSubframes;
ue.NFrame = a_NFrame;
ue.NSubframe = a_NSubframe;

[puschInd, info] = ltePUSCHIndices(ue,ue.PUSCH);
% PUSCH Indices = info.Gd = 864 = 6 RB * 12 Sub-carrier * 12 symbol
% (horizontal axis) since we are drawing only 1 subframe = 2 slots = 14
% symbols, 2 of the 14 symbols are for PUCCH and the rest are for PUSCH
% Note: info.G = # of total bits = 2 * info.Gd (QPSK symbols) = 1728

% ueDim = lteULResourceGridSize(ue);
% ueDim = 72 x 14 = (6 RBs * 12 sc) * 14 symbols

rng(5);
%%% make the ue send multiple times to have a smooth BER curve
%%% each time the ue send 576 bits 
N_SCFDMA = 20;
 
bits = randi([0,1],N_SCFDMA*info.G/3,ue.PUSCH.NLayers);
%%% divide the bits to number of columns each column takes info.G/3 from
%%% the bits variable
temp = reshape(transpose(bits), info.G/3, N_SCFDMA);
% Generate random (zeros and ones) with size of (1728) x (1 layer - SISO)
%% *Transmitter*
%% *0. Channel Encoder*
for L = 1 : N_SCFDMA
	
	encodded_bits = lteConvolutionalEncode(temp(:,L));
	%% *1. Scrambling*
	
	scrambledBits = lteULScramble(ue,encodded_bits);
	%% *2. Modulation Mapper*
	
	modulatedSymbols = lteSymbolModulate(scrambledBits,ue.PUSCH.Modulation);
	%% *3. UL Transform Precoder*
	
	precodedSymbols = lteULPrecode(modulatedSymbols,a_NULRB_UE);
	%% *4. RE Mapper*
	
	% Generate resource mapping grid, populate the grid with the precoded symbols
	%% grid is in the form of 100 RBs * 14 OFDM symbols (one subframe)
	% each RB contains 12 subcarriers --> each row represent a subcarrier
	% therefore, grid = 100*12 rows  * 14 columns
	txGrid = lteULResourceGrid(ue);
	txGrid(puschInd) = precodedSymbols;
	%% *5. SC-FDMA Signal Generator*
	
	% Performing SC-FDMA modulation
	[txTimeDomainSigtemp,~] = lteSCFDMAModulate(ue,txGrid);
	txTimeDomainSig(:,L) = txTimeDomainSigtemp;
	
end

%% *Adding AWGN*

SNR = [-5:1:5];
BER = zeros(1,length(SNR));
theoBER = zeros(1,length(SNR));
for i = 1 : 1 : length(SNR)
%     N0 = 0.5 / (10^(SNR(i)/10));
%     N = sqrt(N0/2) / sqrt(double(2048));
% 	%N = sqrt(N0/2) * sqrt(100 * 12);
	SNR2 = 10^(SNR(i)/20);
    N = 1/(SNR2*sqrt(2*double(2048)));
    % Add noise
    noise = N*complex(randn(size(txTimeDomainSig)), randn(size(txTimeDomainSig)));
     txSignalWithNoise = txTimeDomainSig + noise;
% 	 for L = 1 : N_SCFDMA
% 		 txSignalWithNoise(:,L) = awgn(txTimeDomainSig(:,L), SNR(i),'measured');
% 	 end
%% *Receiver*
%% *5. SC-FDMA Demodulation*
	%%% loop until all the transmitted bits (N_SCFDMA*info.G/3) are decoded
	%%% each loop decode (info.G/3) bits
	decoddedBits = [];
	for L = 1 : N_SCFDMA
		% Performing SC-FDMA Demodulation
		demodGrid = lteSCFDMADemodulate(ue,txSignalWithNoise(:,L));
		%% *4. RE Demapper*
		
		demappedGrid = demodGrid(puschInd);
		%% *3. UL Transform Decoder*
		
		decodedSymbols = lteULDeprecode(demappedGrid,a_NULRB_UE);
		%% *2. Modulation Demapper*
		
		%%% making a hard decision in receiver, then convert it to soft
		%%% bits. the next functions work on soft bits only	
		demodSymbols = lteSymbolDemodulate(decodedSymbols,ue.PUSCH.Modulation,'Hard') - 0.5;
		
		%%% to simulate soft decision bit error rate, uncomment this line
		% demodSymbols = lteSymbolDemodulate(decodedSymbols,ue.PUSCH.Modulation);
		
		%% *1. Descrambling*
		
		descrambledBits = lteULDescramble(ue,demodSymbols);
		%descrambledBits = demodSymbols;
		%% *0. Channel Decoder
		decoddedBitstemp = lteConvolutionalDecode(descrambledBits);
		decoddedBits = [decoddedBits; decoddedBitstemp];
	end
		% Converting bits to non-polar again
		rxBits = decoddedBits;
		rxBits(rxBits<0) = 0;
		rxBits(rxBits>0) = 1;
		% rxBits = (round(descrambledBits) + 1) / 2;
		BER(i) = sum(abs(cast(rxBits,'double')-bits))/ length(bits);
		theoBER(i) = berawgn(SNR(i),'psk',4,'nondiff');
	
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
