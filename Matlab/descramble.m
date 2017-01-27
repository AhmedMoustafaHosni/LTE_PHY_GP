% Function:    descrambler
% Description: descramble bits with psuedo random seq.
% Inputs:      b         - data bits
%              c         - psuedo random sequence
% Outputs:     b_tilde   - descrambled bits
%edit 27/1/2017

function b_tilde = descramble(c,b)

% generate the descrambled bits
b_tilde = mod(b+c,2);

end
