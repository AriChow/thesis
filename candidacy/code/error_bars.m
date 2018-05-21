best_errs = [];
best_errs_std = [];
best_configs = {};

for i = 1 : 5
    for j = 1 : 7
        for k = 1 : 4
            best_errs = [best_errs, matsc_results(i, j, k)];
            best_errs_std = [best_errs_std, matsc_results_std(i, j, k)];
            best_configs = [best_configs; [{fe{i}}, {fs{j}}, {clf{k}}]];
        end
    end
end

