% Function:    cmn_dec2bin
% Description: Converts an array of decimal numbers
%              to an array of binary arrays
% Inputs:      dec      - Array of decimal numbers
%              num_bits - Number of bits per decimal
%                         number
% Outputs:     array    - Array of binary arrays


function [array] = cmn_dec2bin(dec, num_bits)
[junk, num_dec] = size(dec);

for(n=1:num_dec)
	tmp = dec(n);
	for(m=num_bits-1:-1:0)
		array(n,num_bits-m) = floor(tmp/2^m);
		tmp                 = tmp - floor(tmp/2^m)*2^m;
	end
end
end
