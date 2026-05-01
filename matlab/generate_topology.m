function [adjMatrix, delayMatrix, positions] = generate_topology(params, epochOffset)
% GENERATE_TOPOLOGY  Build LEO ISL topology for one or more consecutive epochs.
%
% Called by the Python bridge (MatlabTopologyBridge) via:
%   - matlab.engine (Strategy 1)
%   - generate_topology_cli wrapper (Strategy 2)
%   - MATLAB Production Server (Strategy 3)
%
% When params.numEpochs > 1 the function builds the satellite scenario ONCE
% and steps through each epoch snapshot, returning 3-D arrays [S×S×E] for
% adjMatrix / delayMatrix and [S×3×E] for positions.  When numEpochs == 1
% (default) the outputs are squeezed back to 2-D [S×S] / [S×3] so that all
% existing single-epoch callers continue to work without modification.
%
% Inputs
% ------
%   params      struct with fields:
%     .totalSats        total number of satellites
%     .numPlanes        number of orbital planes (Walker-Star)
%     .phasing          Walker phasing parameter
%     .altitudeKm       orbit altitude in km above Earth surface
%     .inclinationDeg   orbit inclination in degrees
%     .islModel         'nearest_k' | 'grid'
%     .islK             max ISL neighbours per satellite (nearest_k)
%     .islMaxRangeKm    maximum ISL range in km
%     .epochStartUTC    start time string, e.g. '2025-01-01 00:00:00'
%     .epochOffsetS     seconds from epochStartUTC for the FIRST snapshot
%     .epochDurationS   duration of one epoch in seconds
%     .numEpochs        (optional, default 1) number of consecutive epochs
%
%   epochOffset  (scalar, seconds) — alias for params.epochOffsetS,
%                accepted as a second positional arg for MPS compatibility.
%
% Outputs
% -------
%   adjMatrix   [S×S] or [S×S×E]   : binary adjacency (1 = ISL exists)
%   delayMatrix [S×S] or [S×S×E]   : one-way ISL propagation delay (ms)
%   positions   [S×3] or [S×3×E]   : [lat_deg, lon_deg, alt_km]
%
% Requirements
% ------------
%   MATLAB R2021a+, Satellite Communications Toolbox

    % ── 0. Resolve inputs ────────────────────────────────────────────────
    if nargin >= 2 && ~isempty(epochOffset)
        params.epochOffsetS = epochOffset;
    end
    if ~isfield(params, 'epochOffsetS')
        params.epochOffsetS = 0;
    end
    numEpochs = 1;
    if isfield(params, 'numEpochs')
        numEpochs = max(1, round(params.numEpochs));
    end

    S       = params.totalSats;
    nPlanes = params.numPlanes;
    EARTH_RADIUS_M = 6.378137e6;

    fprintf('[MATLAB] generate_topology: totalSats=%d, numPlanes=%d, altKm=%.1f, numEpochs=%d\n', ...
        S, nPlanes, params.altitudeKm, numEpochs);

    % ── 1. Build scenario spanning all epochs (created once) ─────────────
    startTime       = datetime(params.epochStartUTC, ...
                               'InputFormat', 'yyyy-MM-dd HH:mm:ss', ...
                               'TimeZone', 'UTC');
    firstSnapOffset = params.epochOffsetS;
    totalSpanS      = firstSnapOffset + numEpochs * params.epochDurationS;
    stopTime        = startTime + seconds(totalSpanS);

    sc = satelliteScenario(startTime, stopTime, 1);   % 1-second sample rate

    orbitalRadiusM = EARTH_RADIUS_M + params.altitudeKm * 1e3;
    constellation  = walkerStar(sc, orbitalRadiusM, params.inclinationDeg, ...
                                S, nPlanes, params.phasing);

    % ── 2. Pre-allocate 3-D output arrays ────────────────────────────────
    adjMatrix   = zeros(S, S, numEpochs);
    delayMatrix = zeros(S, S, numEpochs);
    positions   = zeros(S, 3, numEpochs);

    % ── 3. Loop over each epoch snapshot ─────────────────────────────────
    for e = 1:numEpochs
        epochOffsetE = firstSnapOffset + (e - 1) * params.epochDurationS;
        snapTime     = startTime + seconds(epochOffsetE);

        % --- Satellite positions at this snapshot time ---
        stateData = states(constellation, snapTime, 'CoordinateFrame', 'ecef');
        sz = size(stateData);
        fprintf('[MATLAB] Epoch %d: stateData size = %s\n', e, mat2str(sz));

        if sz(1) == S && sz(2) == 6
            st = stateData(:, :, 1);
        elseif sz(2) == S && sz(3) == 6
            st = squeeze(stateData(1, :, :));
        elseif sz(1) == 6 && sz(2) == S
            st = stateData';
        else
            st = reshape(stateData, S, []);
            if size(st, 2) > 6, st = st(:, 1:6); end
        end

        pos_ecef = st(:, 1:3);   % [S×3] metres

        actualSats = size(pos_ecef, 1);
        if actualSats ~= S
            error('generate_topology:satCountMismatch', ...
                'pos_ecef has %d rows but params.totalSats=%d (epoch %d).', ...
                actualSats, S, e);
        end

        [lat_deg, lon_deg, alt_m] = ecef2geodetic(wgs84Ellipsoid('meter'), ...
                                        pos_ecef(:,1), pos_ecef(:,2), pos_ecef(:,3));
        positions(:,:,e) = [lat_deg, lon_deg, alt_m / 1e3];

        % --- Distance matrix [S×S] in km ---
        if license('test', 'Statistics_Toolbox')
            distKm = pdist2(pos_ecef, pos_ecef) / 1e3;
        else
            diff3  = permute(pos_ecef, [1 3 2]) - permute(pos_ecef, [3 1 2]);
            distKm = sqrt(sum(diff3.^2, 3)) / 1e3;
        end

        % --- Build ISL adjacency for this epoch ---
        [adjE, delayE] = build_isl(distKm, S, nPlanes, params);
        adjMatrix(:,:,e)   = adjE;
        delayMatrix(:,:,e) = delayE;

        nLinks   = sum(adjE(:)) / 2;
        dVals    = delayE(delayE > 0);
        avgDelay = 0;
        if ~isempty(dVals), avgDelay = mean(dVals); end
        fprintf('[MATLAB] Epoch %d (+%.0fs) | ISL links=%d | avg_delay=%.2f ms\n', ...
            e, epochOffsetE, nLinks, avgDelay);
    end

    % ── 4. Squeeze to 2-D for single-epoch backward compatibility ────────
    if numEpochs == 1
        adjMatrix   = adjMatrix(:,:,1);
        delayMatrix = delayMatrix(:,:,1);
        positions   = positions(:,:,1);
    end
end


% ── Local helper: build ISL adjacency from a distance matrix ─────────────

function [adjOut, delayOut] = build_isl(distKm, S, nPlanes, params)
    adjOut   = zeros(S, S);
    delayOut = zeros(S, S);
    SPEED_OF_LIGHT_KM_S = 2.998e5;

    switch lower(params.islModel)

        case 'nearest_k'
            K        = params.islK;
            maxRange = params.islMaxRangeKm;
            for i = 1:S
                dRow       = distKm(i, :);
                dRow(i)    = Inf;
                dRow(dRow > maxRange) = Inf;
                [~, idx]   = sort(dRow);
                nFeasible  = sum(~isinf(dRow));
                neighbors  = idx(1:min(K, nFeasible));
                for j = neighbors
                    adjOut(i,j) = 1;   adjOut(j,i) = 1;
                    d_ms = (distKm(i,j) / SPEED_OF_LIGHT_KM_S) * 1e3;
                    delayOut(i,j) = d_ms;   delayOut(j,i) = d_ms;
                end
            end

        case 'grid'
            satsPerPlane = S / nPlanes;
            maxRange     = params.islMaxRangeKm;
            for plane = 0:(nPlanes-1)
                for pos = 0:(satsPerPlane-1)
                    i      = plane * satsPerPlane + pos + 1;
                    j_prev = plane * satsPerPlane + mod(pos-1, satsPerPlane) + 1;
                    j_next = plane * satsPerPlane + mod(pos+1, satsPerPlane) + 1;
                    for j = [j_prev, j_next]
                        if distKm(i,j) <= maxRange
                            adjOut(i,j) = 1;   adjOut(j,i) = 1;
                            d_ms = (distKm(i,j) / SPEED_OF_LIGHT_KM_S) * 1e3;
                            delayOut(i,j) = d_ms;   delayOut(j,i) = d_ms;
                        end
                    end
                    j_pp = mod(plane-1, nPlanes) * satsPerPlane + pos + 1;
                    j_np = mod(plane+1, nPlanes) * satsPerPlane + pos + 1;
                    for j = [j_pp, j_np]
                        if j ~= i && distKm(i,j) <= maxRange
                            adjOut(i,j) = 1;   adjOut(j,i) = 1;
                            d_ms = (distKm(i,j) / SPEED_OF_LIGHT_KM_S) * 1e3;
                            delayOut(i,j) = d_ms;   delayOut(j,i) = d_ms;
                        end
                    end
                end
            end

        otherwise
            error('generate_topology:unknownModel', ...
                  'Unknown ISL model: %s. Use ''nearest_k'' or ''grid''.', params.islModel);
    end

    assert(issymmetric(adjOut),    'Adjacency matrix must be symmetric');
    assert(issymmetric(delayOut),  'Delay matrix must be symmetric');
    assert(all(diag(adjOut) == 0), 'Self-loops must be zero in adjMatrix');
end
