% Function:    predecoding_mimo
%
% Description: perform predecoding on complex data after transform precoder.
%			   predecoding is to map data streams from antenna ports to layers. possible
%			   number of antenna ports are either 2 or 4.
%
%			   for now, only Full spatial multiplexing is supported:
%			   1- 2 layers with 2 antenna ports
%			   2- 4 layers with 4 antenna ports
%			   
% Inputs:      data         - complex data matrix with dim (P * M_symbs_layer)
%			   N_layers     - number of layers
%			   N_antennas   - number of antenna ports
%
% Outputs:     y            - predecoded data matrix with dim (M_symbs_layer * v)
% where M_symbs_layer is the number of symbols per layer
%              P        is the number of antennas
%			   v        is the number of layers
% created by: Mohammed Osama 

function [y] = predecoding_mimo(data, N_layers, N_antennas)

%% Output = inverse(Precoding_matrix) * data_matrix

% choosing the suitable precoding matrix W depending on number of layers
% and number of antennas. 
if(N_antennas == 2 && N_layers == 2)	
	W = 1/sqrt(2) .* [1 0; 0 1]; 
% The scale multiplied by the matrix is to normalize the power of the matrix.
% the precoding block don't add power to the input stream.
elseif(N_antennas == 4 && N_layers == 4)
	W = 1/2 .* [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
elseif(N_antennas == 64 && N_layers == 64)
	W = 1/8 .* eye(64);
else
	error('Not supported number of antennas or number of layers');
end
M = size(data);
if (M ~= N_layers)
	error('the input matrix has incorrect dimensions (adjust the number of layers)')
end
tmp = inv(W) * data;
y = transpose(tmp);
end
