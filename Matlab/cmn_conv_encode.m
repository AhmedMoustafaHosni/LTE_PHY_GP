% Function:    convolutin_encoding
% Description: Convolutionally encodes the input
%              bit array according to the parameters
% Inputs:      in       - Input bit array
%              r        - Rate
%              g        - Polynomial definition array in octal
%			   k        - constraint length of output stream
% Outputs:     out      - convolutionally encoded bit array
function [out] = convolution_encoding(in, r, g, k)
%%% for testing
% r = 3;
% g = [133 171 165];
% k = 7;
% in = [1 0 0 0 0 0 0 ]; %% 7 input bits
% Initialize shift register with tail bits of input
for n = 0 : k-1  %% from 0 to 6
	%%% eqn from the standard is S(i) = C(K-1-i)
	s_reg(n+1) = in(length(in)-n); % initialize the nth register with the (last_input-n) 
end

for i = 1 : length(g)
	tmp = oct2dec(g(i));
	g_array(i,:) = fliplr(de2bi(tmp));
end 

% Convolutionally encode input
idx = 1;
for n = 0 : length(in)-1
	% Add next bit to shift register
	for m = k : -1 : 2
		s_reg(m) = s_reg(m-1); %%% shift each value in shift registers to the next register
	end
	%%% for loop stops at shifting the value at 1st register to the last register
	%%% add input in the 1st shift reg
	s_reg(1) = in(n+1);
	
	% Determine the output bits
	for m = 0 : r-1
		out(idx) = 0;
		for q = 0 : k-1
			out(idx) = out(idx) + s_reg(q+1)*g_array(m+1,q+1);
		end
		out(idx) = mod(out(idx), 2);
		idx      = idx + 1;
	end
end
end
