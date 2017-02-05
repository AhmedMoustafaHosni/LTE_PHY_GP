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
M_pusch_sc = M_pusch_rb*N_sc_rb;  %% total number o-f subcarriers
Nc = 1600;        %% specified by the 36211 v8 standard for random sequence generation      
n_s = 0;          %% assume UE send on time slot 4
n_RNTI = 10;      %% radio network temporary identifier given to the UE by enodeB (assume 10)


%-------------------------transmitter-----------------------------------

% Interleaver
N_l = 1;                                %Number of Layers
Q_m = 2;                                %QPSK Modulation
ri_bits = [];                           %No RI bits
ack_bits = [];                          %No ACK bits since they replace the original data (to be able to compare
                                        %with the original data after deinterleaving)
                                        
ue.CyclicPrefixUL = 'Normal';
ue.Shortened = 0;
chs.Modulation = 'QPSK';
chs.NLayers = 1;
numRB = 1;
numREperRB = 144;
bitsPerSymbol = 2;
numBits = numRB * numREperRB * bitsPerSymbol;

interleaved_data = lteULSCHInterleave(ue, chs, b, [], [], []);

% mapping

symbs = mapper(interleaved_data, 'qpsk');

% generate sc-fdma signal

 idx = 0;
 N_symbs_per_slot=7;
 for(L=0:(N_symbs_per_slot*2)-1)
	 if(3 == L)
		 for(k=0:M_pusch_sc-1)
			 % DMRS
			 mod_vec_out(L+1,k+1) = 0;
		 end
	 elseif(10 == L)
		 for(k=0:M_pusch_sc-1)
			 % DMRS
			 %mod_vec_out(L+1,k+1) = r2(M_pusch_sc+k+1); %%%%%%%%%%%% 
			 mod_vec_out(L+1,k+1) = 0;
		 end
	 else
		 for(k=0:M_pusch_sc-1)
			 % PUSCH
			 mod_vec_out(L+1,k+1) = symbs(idx+1);
			 idx                  = idx + 1;
		 end
	 end
 end

pusch_bb = sc_fdma_modulator(mod_vec_out , M_pusch_rb);

txTimeDomainSig = pusch_bb;

%% *Adding AWGN*

SNR = (-2:1:5);
BER = zeros(1,length(SNR));
theoBER = zeros(1,length(SNR));
for i = 1 : 1 : length(SNR)
    N0 = 0.5 / (10^(SNR(i)/10));
    N = sqrt(N0/2) / sqrt(double(2048));
    % Add noise
    noise = N*complex(randn(size(txTimeDomainSig)), randn(size(txTimeDomainSig)));
    txSignalWithNoise = txTimeDomainSig + noise;

    %----------------------------reciever------------------------------------

    % sc-fdma demodulation

    rec_symbs = sc_fdma_demodulator(txSignalWithNoise, M_pusch_rb);
    
    % mapping the symbols to one vector

    rec_symbs_one_vec = reshape(rec_symbs.',1,72*14);
    rec_symbs_no_DMRs = [rec_symbs_one_vec(1:3*72) rec_symbs_one_vec(4*72+1:10*72) rec_symbs_one_vec(11*72+1:end)]; 
    
    % demapping

    demapped_bits = demapper_hard(rec_symbs_no_DMRs, 'qpsk');

    % deinterleaver
    
    [out_bits,~,~,~] = lteULSCHDeinterleave(ue,chs,demapped_bits);

    BER(i) = sum(abs(out_bits-b))/ double(length(b));
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