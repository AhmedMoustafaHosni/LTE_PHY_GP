% Function:    cmn_oct2bin
% Description: Converts an array of octal numbers
%              to an array of binary arrays
% Inputs:      oct      - Array of octal numbers
%              num_bits - Number of bits per octal
%                         number
% Outputs:     array    - Array of binary arrays



function [array] = cmn_oct2bin(oct, num_bits)
[junk, num_oct] = size(oct);

for(n=1:num_oct)
	% Convert whole digits
	tmp = oct(n);
	idx = num_bits;
	for(m=1:floor(num_bits/3))
		dig                = mod(tmp, 10);
		array(n,idx-2:idx) = cmn_dec2bin(dig, 3);
		tmp                = floor(tmp/10);
		idx                = idx - 3;
	end
	
	if(mod(num_bits, 3) ~= 0)
		% Convert non-whole digits
		dig             = mod(tmp, 10);
		array(n, 1:idx) = cmn_dec2bin(dig, mod(num_bits, 3));
	end
end
end
