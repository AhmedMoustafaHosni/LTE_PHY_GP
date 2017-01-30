in = [ 1 1 1 0 0 0 1 1 1 1];
in1 = convolutional_encoder(in);
in2 = lteConvolutionalEncode(in');
out1 = viterbi_decoder(in1);
out2 = lteConvolutionalDecode(in2);
