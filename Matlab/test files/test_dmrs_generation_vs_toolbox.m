%% Generate PUSCH DM-RS
% Generated the PUSCH Demodulation Reference Signal (DM-RS) values for
% UE-specific settings.

%%
% Initialize UE specific (|ue|) and channel (|chs|) configuration
% structures. Generate PUSCH DM-RS values.
ue.NCellID = 2;
ue.NSubframe = 0;
ue.CyclicPrefixUL = 'Normal';
ue.Hopping = 'Off';
ue.SeqGroup = 0;
ue.CyclicShift = 0;
ue.NTxAnts = 1;

chs.PRBSet = (0:5).';
chs.NLayers = 1;
chs.OrthCover = 'Off';
chs.DynCyclicShift = 0;

for i = 0:9
    ue.NSubframe = i;
    puschSeq(:,i+1) = ltePUSCHDRS(ue,chs);
end

for i = 0:9
    x(:,i+1) = generate_dmrs_pusch(i, 2, 0, 0, 0, 0, 0, 'fixed', 6, 0).';
end

sum(sum(x - puschSeq))