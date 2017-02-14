% Function:    layer_demapping 
% Description: demapping demapped data from different layers
%                        
% Inputs:      layer_demapped_data       - Input baseband mapped data
%              number_of_layers          - number of layers used
% Outputs:     demapped_symbols_x         - Ouput demapped symboles
% Spec:        N/A
% Notes:       None
% Modified by: Khaled Ahmed Ali
function [demapped_symbols_1,demapped_symbols_2,demapped_symbols_3,demapped_symbols_4] =  layer_demapping (layer_demapped_data,number_of_layer)
if(number_of_layer == 1)
    demapped_symbols_1 = layer_demapped_data;
elseif(number_of_layer == 2)
    demapped_symbols_1 = layer_demapped_data(:,1);
    demapped_symbols_2 = layer_demapped_data(:,2);
elseif(number_of_layer == 3)
    demapped_symbols_1 = layer_demapped_data(:,1);
    demapped_symbols_2 = layer_demapped_data(:,2);
    demapped_symbols_3 = layer_demapped_data(:,3);
else
    demapped_symbols_1 = layer_demapped_data(:,1);
    demapped_symbols_2 = layer_demapped_data(:,2);
    demapped_symbols_3 = layer_demapped_data(:,3);
    demapped_symbols_4 = layer_demapped_data(:,4);
end
end