% Function:    scrambler
% Description: scramble bits with psuedo random seq.
% Inputs:      b         - data bits (each layer's data in a row)
%              c         - psuedo random sequence (each layer's data in a row)
%                         (We assumed that every layer has it's own PRS)
%              N_layers  - Number of Layers used in MIMO
% Outputs:     b_tilde   - scrambled bits (each layer's data in a row)
%edit 16/2/2017

function b_tilde = scrambler_MIMO(b, c, N_layers)
    for i = 1:N_layers
        % generate the scrambled bits
        b_tilde(i,:) = mod(b(i,:)+c(i,:),2);
    end
end
