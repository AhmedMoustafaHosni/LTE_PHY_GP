% Function:    convolutional_encoder mimo
% Description: Convolutionally encodes the input
%              bit array according to standard of 
%              release 10
% Inputs:      in                - Input bit array
%              number_of_layers  - Number of layers used
% Outputs:     encoded_bits_1      - Ouput bit array for layer 1
%              encoded_bits_2      - Ouput bit array for layer 2
%              encoded_bits_3      - Ouput bit array for layer 3
%              encoded_bits_4      - Ouput bit array for layer 4
% Spec:        N/A
% Notes:       putting number_of_layers = 1 then it is a siso function
% Modified by: Khaled Ahmed Ali 
function [encoded_bits_1,encoded_bits_2,encoded_bits_3,encoded_bits_4] = convolutional_encoder_mimo(in,number_of_layers)
size = length(in);
encoded_bits_1 = 0;
encoded_bits_2 = 0;
encoded_bits_3 = 0;
encoded_bits_4 = 0;

if(number_of_layers == 4)
    if(size/4 < 7) % number of registers
        error('data used is less than the constraint length K=7');
    end
    encoded_bits_1 = convolutional_encoder(in( 1:size/4 ));
    encoded_bits_2 = convolutional_encoder( in(size/4 +1 : size/2) );
    encoded_bits_3 = convolutional_encoder(in(size/2 +1 : 3*size/4));
    encoded_bits_4 = convolutional_encoder( in(3*size/4 +1 : size));
elseif (number_of_layers == 3)
    if(size/3 < 7) % number of registers
        error('data used is less than the constraint length K=7');
    end
    encoded_bits_1 = convolutional_encoder(in( 1:size/3 ));
    encoded_bits_2 = convolutional_encoder( in(size/3 +1 : 2*size/3 ));
    encoded_bits_3 = convolutional_encoder(in(2*size/3 +1 : size));
elseif (number_of_layers == 2)
    if(size/2 < 7) % number of registers
        error('data used is less than the constraint length K=7');
    end
    encoded_bits_1 = convolutional_encoder(in( 1:size/2 ));
    encoded_bits_2 = convolutional_encoder( in(size/2 +1 : size) );
else
    if(size < 7) % number of registers
        error('data used is less than the constraint length K=7');
    end
    encoded_bits_1 = convolutional_encoder(in);
end
