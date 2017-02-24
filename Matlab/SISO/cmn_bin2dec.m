% Function:    cmn_bin2dec
% Description: Converts an array of binary arrays
%              to an array of decimal numbers
% Inputs:      array    - Array of binary arrays
%              num_bits - Number of bits per decimal
%                         number
% Outputs:     dec      - Array of decimal numbers

function [dec] = cmn_bin2dec(array, num_bits)
    [num_array, junk] = size(array);

    for(n=1:num_array)
        dec(n) = 0;
        for(m=num_bits-1:-1:0)
            dec(n) = dec(n) + array(n,num_bits-m)*2^m;
        end
    end
end
