g = [133 171 165];
r = 3;
in = [1 0 0 0 0 0 0];
k = 7;
encoded_data = cmn_conv_encode(in,r,g,k);
decoded_data = cmn_viterbi_decode(encoded_data,r,g,k);
