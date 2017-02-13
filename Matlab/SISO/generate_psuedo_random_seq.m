% Function:    generate_psuedo_random_seq
% Description: Generates the psuedo random sequence c
% Inputs:      c_init    - Initialization value for the sequence
%              seq_len   - Length of the output sequence
% Outputs:     c         - Psuedo random sequence
%edit 27/1/2017

function c = generate_psuedo_random_seq(c_init, seq_len)

%% specified by the 36211 v8 standard for random sequence generation      
Nc = 1600;  

% the first sequence x1
x1 = zeros (1,Nc + seq_len);
x1(1) = 1;  %% first value in x1 is initialized to 1 according to the standard.

% the second sequence x2
x2 = zeros (1,Nc + seq_len);

% initialization of x2 depending on cinit
x2_init = de2bi(c_init);

% initializing the second sequence with the binary representation of cinit  
x2(1:length(x2_init)) =  x2_init;

% generate the remaining part of x1 and x2 sequences
for n = 1 : Nc+seq_len-31
        x1(n+31) = mod(x1(n+3) + x1(n), 2);
        x2(n+31) = mod(x2(n+3) + x2(n+2) + x2(n+1) + x2(n), 2);
end

% Generate c(n)
c = zeros (1,seq_len);
for n = 1 : seq_len
    c(n) = mod(x1(n+Nc) + x2(n+Nc), 2);
end

end
