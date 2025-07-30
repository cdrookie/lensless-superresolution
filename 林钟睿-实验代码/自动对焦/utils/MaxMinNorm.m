function norm_res = MaxMinNorm(x,bound)

coefficient = bound(2)-bound(1);

max_value = max(x(:));
min_value = min(x(:));

norm_res = bound(1) + coefficient*(x - min_value)/(max_value-min_value); 