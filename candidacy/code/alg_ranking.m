clear
load('/Users/aritrachowdhury/Documents/research/thesis/candidacy/results/sorted_trans_long_error_bars.mat')
algs = T{:, 1};
fe_map = containers.Map;
dr_map = containers.Map;
la_map = containers.Map;
for i = 1 : length(algs)
    alg = algs{i};
    a = split(alg, ',');
    a = {strip(a{1}), strip(a{2}), strip(a{3})};
    if isKey(fe_map, a{1})
        fe_map(a{1}) = [fe_map(a{1}), i];
    else
        fe_map(a{1}) = [i];
    end
    if isKey(dr_map, a{2})
        dr_map(a{2}) = [dr_map(a{2}), i];
    else
        dr_map(a{2}) = [i];
    end
    if isKey(la_map, a{3})
        la_map(a{3}) = [la_map(a{3}), i];
    else
        la_map(a{3}) = [i];
    end
end

k = keys(fe_map);
v = values(fe_map);
for i = 1 : length(k)
    fe_map(k{i}) = mean(v{i});
end
T_fe = table(values(fe_map)', 'RowNames', keys(fe_map), 'VariableNames', {'Rank'});

k = keys(dr_map);
v = values(dr_map);
for i = 1 : length(k)
    dr_map(k{i}) = mean(v{i});
end
T_dr = table(values(dr_map)', 'RowNames', keys(dr_map), 'VariableNames', {'Rank'});

k = keys(la_map);
v = values(la_map);
for i = 1 : length(k)
    la_map(k{i}) = mean(v{i});
end
T_la = table(values(la_map)', 'RowNames', keys(la_map), 'VariableNames', {'Rank'});