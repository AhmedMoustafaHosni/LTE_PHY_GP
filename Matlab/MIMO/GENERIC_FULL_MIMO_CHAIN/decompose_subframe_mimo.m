% Function:    decompose_subframe_mimo
% Description: decompose the subframe 
% Inputs:      subframe_per_ant      - complex data to be decoded from 3D
%			   matrix of size (M_sc_pusch * 14(M_symbols_subframe) * P)
% where P is the antenna ports
%       M is the reference symbols
% Outputs:     Decoded_streams       - Matrix of dimensions (P * M_symbs_sent_per_ant)
% Add MIMO by Khaled Ahmed and Mohammed Osama 

function [Decoded_streams] = decompose_subframe_mimo(subframe_per_ant, Ant_Ports)


for P = 1 : Ant_Ports

	Decoded_streams(P,:) = [subframe_per_ant(:,1,P).' subframe_per_ant(:,2,P).' subframe_per_ant(:,3,P).'  subframe_per_ant(:,5,P).' subframe_per_ant(:,6,P).' subframe_per_ant(:,7,P).' subframe_per_ant(:,8,P).' subframe_per_ant(:,9,P).' subframe_per_ant(:,10,P).' subframe_per_ant(:,12,P).' subframe_per_ant(:,13,P).' subframe_per_ant(:,14,P).'];

end
end