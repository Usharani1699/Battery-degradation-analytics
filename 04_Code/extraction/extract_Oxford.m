%% Oxford Battery Degradation Dataset — Extract to CSV
% Produces:
%   Extracted_CSV_Data/Oxford/Oxford_CycleSummary.csv          (SoH per cell per char-cycle)
%   Extracted_CSV_Data/Oxford/Oxford_1C_DischargeSeries.csv    (full timeseries)
%
% Run via: matlab -batch "run('extract_Oxford.m')"
% NOTE: The .mat file is 262 MB — loading takes ~1 minute.

clc; clear; close all;

out_dir  = 'C:\Users\pure ev\Music\Term 3\Extracted_CSV_Data\Oxford';
if ~exist(out_dir,'dir'), mkdir(out_dir); end

mat_file = 'C:\Users\pure ev\Music\Term 3\NASA Prognostics (PCoE) Battery Dataset\Oxford\Oxford_Battery_Degradation_Dataset_1.mat';
fprintf('Loading Oxford dataset (262 MB) — please wait ...\n');
S = load(mat_file);
fprintf('Loaded.\n');

nom_cap = 740;   % mAh — Kokam SLPB533459H4

% ── Find cell variables ──────────────────────────────────────────────────────
all_fields = fieldnames(S);
% Typical names: cell1, cell2 ... cell8  OR  Cell1 ... or similar
cell_fields = all_fields(~cellfun('isempty', regexpi(all_fields, '^cell\d+$', 'once')));
if isempty(cell_fields)
    % Try alternate naming
    cell_fields = all_fields(~cellfun('isempty', regexp(all_fields, '\d', 'once')));
end
fprintf('Found %d cell variables: %s\n', numel(cell_fields), strjoin(cell_fields,', '));

sum_rows = cell(0,5);     % summary
ts_rows  = cell(0,7);     % timeseries

for ci = 1:numel(cell_fields)
    cname     = cell_fields{ci};
    cell_data = S.(cname);

    char_cyc_names = fieldnames(cell_data);
    char_cyc_names = sort(char_cyc_names);   % cyc0100, cyc0200, ...
    fprintf('  %s: %d characterisation cycles\n', cname, numel(char_cyc_names));

    for cc = 1:numel(char_cyc_names)
        ccname = char_cyc_names{cc};
        % Extract numeric cycle number, e.g. cyc0100 -> 100
        num_str = regexprep(ccname, '[^0-9]', '');
        if isempty(num_str), cycle_num = NaN; else, cycle_num = str2double(num_str); end

        cyc = cell_data.(ccname);

        % ── 1-C discharge capacity ──────────────────────────────────────────
        c1dc_cap = NaN;
        if isfield(cyc,'C1dc') && isfield(cyc.C1dc,'q') && ~isempty(cyc.C1dc.q)
            c1dc_cap = max(abs(cyc.C1dc.q));
        end
        soh = c1dc_cap / nom_cap * 100;

        % ── OCV discharge capacity ──────────────────────────────────────────
        ocv_cap = NaN;
        if isfield(cyc,'OCVdc') && isfield(cyc.OCVdc,'q') && ~isempty(cyc.OCVdc.q)
            ocv_cap = max(abs(cyc.OCVdc.q));
        end

        sum_rows(end+1,:) = {cname, cycle_num, c1dc_cap, ocv_cap, soh};

        % ── 1-C discharge timeseries ─────────────────────────────────────────
        if isfield(cyc,'C1dc')
            d = cyc.C1dc;
            if isfield(d,'t') && ~isempty(d.t)
                t_v = d.t(:);
                v_v = d.v(:);
                q_v = d.q(:);
                T_v = d.T(:);
                np  = min([numel(t_v), numel(v_v), numel(q_v), numel(T_v)]);
                for ti = 1:np
                    ts_rows(end+1,:) = {cname, cycle_num, 'C1dc', t_v(ti), v_v(ti), q_v(ti), T_v(ti)};
                end
            end
        end
    end
end

% ── Write summary ────────────────────────────────────────────────────────────
if ~isempty(sum_rows)
    T = cell2table(sum_rows,'VariableNames', ...
        {'Cell_ID','Char_Cycle_Num','C1_Discharge_mAh','OCV_Discharge_mAh','SoH_pct'});
    writetable(T, fullfile(out_dir,'Oxford_CycleSummary.csv'));
    fprintf('Saved Oxford_CycleSummary.csv (%d rows)\n', height(T));
end

% ── Write timeseries ─────────────────────────────────────────────────────────
if ~isempty(ts_rows)
    T = cell2table(ts_rows,'VariableNames', ...
        {'Cell_ID','Char_Cycle_Num','Cycle_Type','Time_s','Voltage_V','Charge_mAh','Temperature_C'});
    writetable(T, fullfile(out_dir,'Oxford_1C_DischargeSeries.csv'));
    fprintf('Saved Oxford_1C_DischargeSeries.csv (%d rows)\n', height(T));
end

fprintf('\n=== Oxford extraction COMPLETE ===\n');
