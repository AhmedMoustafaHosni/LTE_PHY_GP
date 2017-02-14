% Function:    viterbi_decoder_mimo
% Description: Viterbi decodes a convolutionally
%              coded input bit array using the
%              provided parameters
% Inputs:      in_x       - Input array x
%              nol        - number of layers
% Outputs:     out        - Ouput bit array
% Spec:        N/A
% Notes:       N/A
% Modified by: Khaled Ahmed Ali 
function [out] = viterbi_decoder_mimo(in_1, in_2, in_3, in_4, nol)
if(nol == 4)
    out = [viterbi_decoder(in_1); viterbi_decoder(in_2) ;viterbi_decoder(in_3);viterbi_decoder(in_4)] ;
elseif(nol == 3)
    out = [viterbi_decoder(in_1); viterbi_decoder(in_2) ;viterbi_decoder(in_3)];
elseif(nol == 2)
    out = [viterbi_decoder(in_1); viterbi_decoder(in_2)];
else
    out = [viterbi_decoder(in_1)];
end