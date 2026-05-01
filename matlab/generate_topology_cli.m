function generate_topology_cli(paramsJsonPath)
% GENERATE_TOPOLOGY_CLI  Command-line wrapper for Strategy 2 (subprocess).
%
% Called by Python as:
%   matlab -batch "generate_topology_cli('/tmp/xyz/params.json')"
%
% Reads params.json, calls generate_topology() for ALL requested epochs in
% one shot, and saves the result to the .mat path in params.result_file.
%
% When params.numEpochs > 1 the saved arrays are 3-D: [S×S×E] for
% adj_matrix / delay_matrix and [S×3×E] for positions.  When numEpochs==1
% (default) the arrays are 2-D (backward compatible with existing callers).

    % ── Read params ───────────────────────────────────────────────────────
    rawText = fileread(paramsJsonPath);
    params  = jsondecode(rawText);

    numEpochs   = 1;
    if isfield(params, 'numEpochs')
        numEpochs = max(1, round(params.numEpochs));
    end
    epochOffset = params.epochOffsetS;

    % ── Generate all epochs in one call ───────────────────────────────────
    [adjMatrix, delayMatrix, positions] = generate_topology(params, epochOffset);

    % ── Save result ───────────────────────────────────────────────────────
    resultFile   = params.result_file;
    adj_matrix   = adjMatrix;    %#ok<NASGU>
    delay_matrix = delayMatrix;  %#ok<NASGU>
    % numEpochs saved so the Python reader knows whether arrays are 2-D or 3-D
    save(resultFile, 'adj_matrix', 'delay_matrix', 'positions', 'numEpochs', '-v7');

    fprintf('[MATLAB CLI] Saved %d epoch(s) to %s\n', numEpochs, resultFile);
end
