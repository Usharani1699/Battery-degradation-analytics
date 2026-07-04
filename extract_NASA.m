%% NASA PCoE Battery Dataset — Extract to CSV
% Produces:
%   Extracted_CSV_Data/NASA/NASA_CycleSummary.csv        (one row per cycle)
%   Extracted_CSV_Data/NASA/B0005_DischargeSeries.csv    (raw timeseries)
%   Extracted_CSV_Data/NASA/NASA_ImpedanceSummary.csv    (Re, Rct per impedance cycle)
%
% Run via: matlab -batch "run('extract_NASA.m')"

clc; clear; close all;

out_dir = 'C:\Users\pure ev\Music\Term 3\Extracted_CSV_Data\NASA';
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

mat_dir = 'C:\Users\pure ev\Music\Term 3\NASA Prognostics (PCoE) Battery Dataset\5. Battery Data Set\1. BatteryAgingARC-FY08Q4';
bat_ids  = {'B0005','B0006','B0007','B0018'};
nom_cap  = 2.0;   % Ah nominal

% ── Accumulators ───────────────────────────────────────────────────────────
sum_rows = cell(0, 7);   % cycle summary
imp_rows = cell(0, 6);   % impedance summary

for bi = 1:numel(bat_ids)
    bname = bat_ids{bi};
    fpath = fullfile(mat_dir, [bname '.mat']);
    if ~isfile(fpath)
        fprintf('SKIP: %s not found\n', fpath); continue;
    end
    fprintf('Loading %s ...\n', bname);
    S = load(fpath);
    battery = S.(bname);
    cycles  = battery.cycle;
    n_cyc   = numel(cycles);
    fprintf('  %d cycles found\n', n_cyc);

    ts_rows = cell(0, 7);   % timeseries rows for this battery
    disch_n = 0;

    for ci = 1:n_cyc
        c   = cycles(ci);
        typ = strtrim(c.type);

        % Ambient temperature
        if isfield(c,'ambient_temperature') && ~isempty(c.ambient_temperature)
            amb = c.ambient_temperature;
        else
            amb = NaN;
        end

        if strcmp(typ,'discharge')
            disch_n = disch_n + 1;

            % Capacity
            if isfield(c.data,'Capacity') && ~isempty(c.data.Capacity)
                cap = max(c.data.Capacity);
            else
                cap = NaN;
            end
            soh = cap / nom_cap * 100;

            sum_rows(end+1,:) = {bname, ci, disch_n, 'discharge', amb, cap, soh};

            % Timeseries
            t_v = c.data.Time(:);
            v_v = c.data.Voltage_measured(:);
            i_v = c.data.Current_measured(:);
            T_v = c.data.Temperature_measured(:);
            np  = min([numel(t_v), numel(v_v), numel(i_v), numel(T_v)]);
            for ti = 1:np
                ts_rows(end+1,:) = {bname, ci, disch_n, t_v(ti), v_v(ti), i_v(ti), T_v(ti)};
            end

        elseif strcmp(typ,'charge')
            sum_rows(end+1,:) = {bname, ci, NaN, 'charge', amb, NaN, NaN};

        elseif strcmp(typ,'impedance')
            d = c.data;
            re  = NaN; rct = NaN;
            if isfield(d,'Re')  && ~isempty(d.Re),  re  = d.Re(end);  end
            if isfield(d,'Rct') && ~isempty(d.Rct), rct = d.Rct(end); end
            imp_rows(end+1,:) = {bname, ci, amb, re, rct, disch_n};
            sum_rows(end+1,:) = {bname, ci, NaN, 'impedance', amb, NaN, NaN};
        end
    end

    % Write per-battery timeseries
    if ~isempty(ts_rows)
        T = cell2table(ts_rows, 'VariableNames', ...
            {'Battery_ID','Raw_Cycle_Idx','Discharge_Cycle_Num','Time_s','Voltage_V','Current_A','Temperature_C'});
        ts_file = fullfile(out_dir, [bname '_DischargeSeries.csv']);
        writetable(T, ts_file);
        fprintf('  Saved timeseries: %s (%d rows)\n', ts_file, height(T));
    end
    clear S battery cycles ts_rows
end

% ── Write cycle summary ─────────────────────────────────────────────────────
if ~isempty(sum_rows)
    T = cell2table(sum_rows, 'VariableNames', ...
        {'Battery_ID','Raw_Cycle_Idx','Discharge_Cycle_Num','Cycle_Type','Ambient_Temp_C','Discharge_Capacity_Ah','SoH_pct'});
    writetable(T, fullfile(out_dir,'NASA_CycleSummary.csv'));
    fprintf('Saved NASA_CycleSummary.csv (%d rows)\n', height(T));
end

% ── Write impedance summary ─────────────────────────────────────────────────
if ~isempty(imp_rows)
    T = cell2table(imp_rows, 'VariableNames', ...
        {'Battery_ID','Raw_Cycle_Idx','Ambient_Temp_C','Re_Ohm','Rct_Ohm','After_Discharge_Num'});
    writetable(T, fullfile(out_dir,'NASA_ImpedanceSummary.csv'));
    fprintf('Saved NASA_ImpedanceSummary.csv (%d rows)\n', height(T));
end

fprintf('\n=== NASA extraction COMPLETE ===\n');
