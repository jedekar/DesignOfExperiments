#!/usr/bin/env python3

import random
import itertools

import numpy as np
import scipy
from scipy.stats import f
from scipy.stats import t
from prettytable import PrettyTable


def f_critical(prob, f1, f2):
    return scipy.stats.f.ppf(prob, f1, f2)


def t_critical(prob, df):
    return scipy.stats.t.ppf(prob, df)


def c_critical(prob, f1, f2):
    return 1 / (
        1 + (f2 - 1) / scipy.stats.f.ppf(1 - (1 - prob) / f2, f1, (f2 - 1) * f1)
    )


x_bounds = [[1, 1], [10, 60], [15, 50], [15, 20]]
factors = len(x_bounds)
experiments = 4
samples = 2
confidence_prob = 0.9

combinations = list(itertools.product([-1, 1], repeat=4))
xn = [combinations[8], combinations[11], combinations[13], combinations[14]]
x = [
    [
        min(x_bounds[j]) if xn[i][j] < 0 else max(x_bounds[j])
        for j in range(len(x_bounds))
    ]
    for i in range(len(xn))
]

y_bounds = [
    int(200 + np.mean([min(x_bounds[i]) for i in range(1, factors)])),
    int(200 + np.mean([max(x_bounds[i]) for i in range(1, factors)])),
]


def create_matrix():
    table = PrettyTable()
    table_head = ["Experiment #"]
    for i in range(factors):
        table_head.append(f"x{i}")

    for i in range(samples):
        table_head.append(f"y{i+1}")

    table.field_names = table_head

    for i in range(experiments):
        table.add_row([i + 1, *xn[i], *y[i]])

    return table


# Cochran's C test

while True:
    y = [
        [random.randint(min(y_bounds), max(y_bounds)) for i in range(samples)]
        for j in range(experiments)
    ]
    matrix = create_matrix()
    
    s2_y = [np.var(y[i]) for i in range(experiments)]
    stat_c = max(s2_y) / sum(s2_y)
    crit_c = c_critical(confidence_prob, samples - 1, experiments)

    print(matrix)
    print(f"Calculated C statistics: {round(stat_c, 3)}")
    print(
        f"Critical C for confidence probability of {confidence_prob}: {round(crit_c, 3)}"
    )

    if stat_c < crit_c:
        print("Variances are equal.")
        break

    print("Variances are not equal. Increasing sample size...")
    samples += 1

my = [np.mean(y[i]) for i in range(len(y))]

xn_col = np.array(list(zip(*xn)))
beta = [np.mean(my * xn_col[i]) for i in range(experiments)]

yn = [sum(beta * np.array(xn[i])) for i in range(experiments)]

print(f"Means: {[round(my[i], 3) for i in range(experiments)]}")
print(f"Calculated function: {[round(yn[i], 3) for i in range(experiments)]}")

delta_x = [abs(x_bounds[i][0] - x_bounds[i][1]) / 2 for i in range(len(x_bounds))]
x0 = [(x_bounds[i][0] + x_bounds[i][1]) / 2 for i in range(len(x_bounds))]
b = [beta[0] - sum(beta[i] * x0[i] / delta_x[i] for i in range(1, factors))]
b.extend([beta[i] / delta_x[i] for i in range(1, factors)])

# Student's t test

s2_b = np.mean(s2_y)
s_beta = np.sqrt(s2_b / samples / experiments)
stat_t = [abs(beta[i]) / s_beta for i in range(factors)]

crit_t = t_critical(confidence_prob, (samples - 1) * experiments)

print(f"Calculated t statistics: {[round(stat_t[i], 3) for i in range(len(stat_t))]}")
print(f"Critical t for confidence probability of {confidence_prob}: {round(crit_t, 3)}")

significant_coeffs = 0
for i in range(len(stat_t)):
    if stat_t[i] < crit_t:
        b[i] = 0
        significant_coeffs += 1

print(f"Regression coefficients: {[round(b[i], 3) for i in range(len(b))]}")

y_calc = [sum((b * np.array(x))[i]) for i in range(experiments)]
print(
    f"Calculated values of model: {[round(y_calc[i], 3) for i in range(len(y_calc))]}"
)

# Fisher's F test

s2_adeq = (
    samples
    / (experiments - significant_coeffs)
    * sum([(y_calc[i] - my[i]) ** 2 for i in range(experiments)])
)

stat_f = s2_adeq / s2_b
crit_f = f_critical(
    confidence_prob, (samples - 1) * experiments, experiments - significant_coeffs
)

print(f"Calculated F statistics: {round(stat_f, 3)}")
print(f"Critical F for confidence probability of {confidence_prob}: {round(crit_f, 3)}")

if stat_f > crit_f:
    print("Model is inadequate.")
else:
    print("Model is adequate.")
