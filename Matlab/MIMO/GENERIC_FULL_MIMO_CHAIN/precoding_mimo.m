% Function:    precoding_mimo
%
% Description: perform precoding on complex data after transform precoder.
%			   precoding is to map layers to antenna ports. possible number of antenna
%			   ports are either 2 or 4. for example: 1 layer (1 data stream) mapped to 2
%			   antenna ports represents "spatial diversity". 2 layers mapped to 2
%			   antenna ports represents "spatial multiplexing". It can be
%			   used to have both "diversity and multiplexing" by choosing 2
%			   layers and 4 antenna ports for example.
%
%			   for now, only Full spatial multiplexing is supported:
%			   1- 2 layers with 2 antenna ports
%			   2- 4 layers with 4 antenna ports
%			   
% Inputs:      data         - complex data matrix with dim (M_symbs_layer * v)
%			   N_layers     - number of layers
%			   N_antennas   - number of antenna ports
%
% Outputs:     z            - precodded data matrix with dim (P * M_symbs_layer)
%							  each row represents the stream that will be sent on each antenna
% where M_symbs_layer is the number of symbols per layer
%            P        is the number of antennas
%			 v        is the number of layers
% REF:      3GPP TS 36.211 section 5.3.3A Release 10
% created by: Mohammed Osama 

function [z] = precoding_mimo(data, N_layers, N_antennas)

%% Output = Precoding_matrix * data_matrix

% choosing the suitable precoding matrix W depending on number of layers
% and number of antennas. 
if(N_antennas == 2 && N_layers == 2)	
	W = 1/sqrt(2) .* [1 0; 0 1]; 
% The scale multiplied by the matrix is to normalize the power of the matrix.
% the precoding block don't add power to the input stream.
elseif(N_antennas == 4 && N_layers == 4)
	W = 1/2 .* [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
elseif(N_antennas == 64 && N_layers == 64)
	W = 1/8 .* eye( 64 ) ;
else
	error('Not supported number of antennas or number of layers');
end
M = size(data);
if (M ~= N_layers)
	error('the input matrix has incorrect dimensions (adjust the number of layers)')
end
z = W * transpose(data);
end
