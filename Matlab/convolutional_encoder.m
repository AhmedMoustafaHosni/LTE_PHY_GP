% Function:    convolutional_encoder 
% Description: Convolutionally encodes the input
%              bit array according to standard of 
%              release 8, to meet matlab toolbox
% Inputs:      in       - Input bit array
% Outputs:     out      - Ouput bit array
% Spec:        N/A
% Notes:       None
% Modified by: Khaled Ahmed Ali & Mohammed Osama
function [encoded_bits] = convolutional_encoder(in)
g_array = [1 0 1 1 0 1 1; 1 1 1 1 0 0 1 ; 1 1 1 0 1 0 1];
r = 3;
k = 7;

% Tail bitting initialization
for(n=0:k-1)
    s_reg(n+1) = in(length(in)-n-1+1); %check page 12 section 5.1.3.1 in 136 212 pdf
end

% Convolutionally encode input
idx = 1;
for(n=0:length(in)-1)
    % Add next bit to shift register
    for(m=k:-1:2)
        s_reg(m) = s_reg(m-1);
    end
    s_reg(1) = in(n+1);
    
    % Determine the output bits
    for(m=0:r-1)
        out(idx) = 0;
        
        for(o=0:k-1)
            out(idx) = out(idx) + s_reg(o+1)*g_array(m+1,o+1);
        end
        out(idx) = mod(out(idx), 2);
        idx      = idx + 1;
    end
end
out = out';
out1 = reshape(out,3,length(out)/3)';
out2 = reshape(out1,1,length(out))';
encoded_bits = out2;
% for test
% out2_b = lteConvolutionalEncode(in);
end