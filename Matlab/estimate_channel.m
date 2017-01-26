% Function:    estimate_channel
% Description: Generates channel estimation 
% Inputs:      symb_0                - received DMRS number 1
%              symb_1                - received DMRS number 2
%              dmrs_0                - generated DMRS number 1
%              dmrs_1                - generated DMRS number 2
%              M_pusch_sc            - number of subcarriers allocated to ue
%              N_symbs_per_slot      - number of symbols per slot
% Outputs:     channel - channel estimation matrix to be used for equalization

%edit: 25/1/2017
%By  : Ahmed Moustafa

function channel = estimate_channel(symb_0, symb_1, dmrs_0, dmrs_1, M_pusch_sc, N_symbs_per_slot)

    % Determine channel estimates
    for m=0:M_pusch_sc-1 
        tmp        = symb_0(m+1)/dmrs_0(m+1);
        mag_0(m+1) = abs(tmp);
        ang_0(m+1) = angle(tmp);

        tmp        = symb_1(m+1)/dmrs_1(m+1);
        mag_1(m+1) = abs(tmp);
        ang_1(m+1) = angle(tmp);
    end

    % Interpolate channel estimates
    for m=0:M_pusch_sc-1
        frac_mag = (mag_1(m+1) - mag_0(m+1))/7;
        frac_ang = ang_1(m+1) - ang_0(m+1);
        if(frac_ang >= pi) % Wrap angle
            frac_ang = frac_ang - 2*pi;
        elseif(frac_ang <= -pi)
            frac_ang = frac_ang + 2*pi;
        end
        frac_ang = frac_ang/7;

        for(L=0:2)
            ce_mag(L+1, m+1)    = mag_0(m+1) - (3-L)*frac_mag;
            ce_ang(L+1, m+1)    = ang_0(m+1) - (3-L)*frac_ang;
            ce_mag(4+L+1, m+1)  = mag_0(m+1) + (1+L)*frac_mag;
            ce_ang(4+L+1, m+1)  = ang_0(m+1) + (1+L)*frac_ang;
            ce_mag(7+L+1, m+1)  = mag_1(m+1) - (3-L)*frac_mag;
            ce_ang(7+L+1, m+1)  = ang_1(m+1) - (3-L)*frac_ang;
            ce_mag(11+L+1, m+1) = mag_1(m+1) + (1+L)*frac_mag;
            ce_ang(11+L+1, m+1) = ang_1(m+1) + (1+L)*frac_ang;
        end
        ce_mag(3+1, m+1)  = mag_0(m+1);
        ce_ang(3+1, m+1)  = ang_0(m+1);
        ce_mag(10+1, m+1) = mag_1(m+1);
        ce_ang(10+1, m+1) = ang_1(m+1);
    end

    % Construct full channel estimates
    for(L=0:(N_symbs_per_slot*2)-1)
        for(m=0:M_pusch_sc-1)
            ce(L+1,m+1) = ce_mag(L+1,m+1)*(cos(ce_ang(L+1,m+1)) + j*sin(ce_ang(L+1,m+1)));
        endfor
    endfor
    
    channel = [ce(0+1,:),    ce(1+1,:),    ce(2+1,:),    ce(4+1,:),    ce(5+1,:),    ce(6+1,:),    ce(7+1,:),    ce(8+1,:),    ce(9+1,:),    ce(11+1,:),    ce(12+1,:),    ce(13+1,:)];
end