% Function:    descrambler
% Description: descramble bits with psuedo random seq.
% Inputs:      b         - data bits
%              c         - psuedo random sequence
% Outputs:     b_tilde   - descrambled bits
%edit 27/1/2017

function b_tilde = descramble(c,b,N_layers)
    for i = 1:N_layers
        % generate the descrambled bits
        b_tilde(i,:) = mod(b(i,:)+c(i,:),2);
    end
end
