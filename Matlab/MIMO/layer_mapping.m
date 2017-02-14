% Function:    layer_mapping 
% Description: mapping mapped data into different layers
%                        
% Inputs:      mapped_symbols_x       - Input baseband mapped data
%              number_of_layers       - number of layers used
% Outputs:     layer_mapped_data      - Ouput layer mapped data
% Spec:        N/A
% Notes:       None
% Modified by: Khaled Ahmed Ali
function [layer_mapped_data] =  layer_mapping (mapped_symbols_1,mapped_symbols_2,mapped_symbols_3,mapped_symbols_4,number_of_layer)
if(number_of_layer == 1)
    layer_mapped_data = [mapped_symbols_1].';
elseif(number_of_layer == 2)
    layer_mapped_data = [mapped_symbols_1; mapped_symbols_2].';
elseif(number_of_layer == 3)
    layer_mapped_data = [mapped_symbols_1 ;mapped_symbols_2; mapped_symbols_3].';
else
    layer_mapped_data = [mapped_symbols_1 ;mapped_symbols_2 ;mapped_symbols_3 ;mapped_symbols_4].';
end
end