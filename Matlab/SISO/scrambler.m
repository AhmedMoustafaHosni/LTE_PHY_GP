% Function:    scrambler
% Description: scramble bits with psuedo random seq.
% Inputs:      b         - data bits
%              c         - psuedo random sequence
% Outputs:     b_tilde   - scrambled bits
%edit 27/1/2017

function b_tilde = scrambler(b, c)

% generate the scrambled bits
b_tilde = mod(b+c,2);

end
