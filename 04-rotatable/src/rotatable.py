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


x_bounds = [np.array([1, 1]),
            np.array([10, 60]),
            np.array([15, 50]),
            np.array([15, 20])]

def extend_inter(a):
    a.extend(
        [a[1] * a[2], a[1] * a[3], a[2] * a[3], a[1] * a[2] * a[3],
         a[1]**2, a[2]**2, a[3]**2]
    )
    return a

x_bounds = extend_inter(x_bounds)

factors = len(x_bounds)
experiments = 15
samples = 3
confidence_prob = 0.99
axial_n = 1.73

y_bounds = [
    int(3.3 + 7.7 * min(x_bounds[1]) + 3.8 * min(x_bounds[2])
    + 1.1 * min(x_bounds[3]) + 2.9 * min(x_bounds[1])
    * min(x_bounds[1]) + 0.5 * min(x_bounds[2]) * min(x_bounds[2])
    + 9.6 * min(x_bounds[3]) * min(x_bounds[3]) + 4.3
    * min(x_bounds[1]) * min(x_bounds[2]) + 0.1 * min(x_bounds[1])
    * min(x_bounds[3]) + 4.9 * min(x_bounds[2]) * min(x_bounds[3])
    + 3.2 * min(x_bounds[1]) * min(x_bounds[2]) * min(x_bounds[3])),
    int(3.3 + 7.7 * max(x_bounds[1]) + 3.8 * max(x_bounds[2])
    + 1.1 * max(x_bounds[3]) + 2.9 * max(x_bounds[1])
    * max(x_bounds[1]) + 0.5 * max(x_bounds[2]) * max(x_bounds[2])
    + 9.6 * max(x_bounds[3]) * max(x_bounds[3]) + 4.3
    * max(x_bounds[1]) * max(x_bounds[2]) + 0.1 * max(x_bounds[1])
    * max(x_bounds[3]) + 4.9 * max(x_bounds[2]) * max(x_bounds[3])
    + 3.2 * max(x_bounds[1]) * max(x_bounds[2]) * max(x_bounds[3]))
]

combinations = list(itertools.product([-1, 1], repeat=4))
xn = combinations[8:]
xn = list(map(list, xn))

for i in range(1, 4):
    axial_comb = [1, 0, 0, 0]
    xn.append([*axial_comb[:i], axial_n, *axial_comb[i+1:]])
    xn.append([*axial_comb[:i], -axial_n, *axial_comb[i+1:]])

xn.extend([[1, *( [0] * 3 )]])

xn = list(map(extend_inter, xn))
xn_col = np.array(list(zip(*xn)))

delta_x = [abs(x_bounds[i][0] - x_bounds[i][1]) / 2 for i in range(len(x_bounds))]
x0 = [(x_bounds[i][0] + x_bounds[i][1]) / 2 for i in range(len(x_bounds))]
x = [[] for i in range(len(xn))]
for i in range(len(xn)):
    for j in range(len(x_bounds)):
        if xn[i][j] == 0:
            x[i].append(x0[j])
        elif xn[i][j] == 1:
            x[i].append(max(x_bounds[j]))
        elif xn[i][j] == -1:
            x[i].append(min(x_bounds[j]))
        elif xn[i][j] > 1:
            x[i].append(axial_n * delta_x[j] + x0[j])
        else:
            x[i].append(-axial_n * delta_x[j] + x0[j])

x_col = np.array(list(zip(*x)))


def create_matrix():
    table = PrettyTable()
    table_head = ["Experiment #"]
    for i in range(factors):
        table_head.append(f"x{i}")

    for i in range(samples):
        table_head.append(f"y{i+1}")

    table.field_names = table_head

    for i in range(experiments):
        table.add_row([i + 1, *np.round(x[i], 3), *y[i]])

    return table

found = False

while not found:
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
    mmy = np.mean(my)

    mxy = list(map(np.mean, x_col * my))
    mxx = [
        [np.mean(x_col[i] * x_col[j]) for j in range(len(x_col))]
        for i in range(len(x_col))
    ]

    equation_matrix = np.array(list(zip(*mxx)))
    constant_terms = mxy

    b = np.linalg.solve(equation_matrix, constant_terms)
    y_test = [sum((b * np.array(x))[i]) for i in range(experiments)]

    print(f"Means: {[round(my[i], 3) for i in range(experiments)]}")
    print(f"Calculated function: {[round(y_test[i], 3) for i in range(experiments)]}")

    # Student's t test
    beta = [sum(my * xn_col[i]) / experiments for i in range(factors)]
    s2_b = np.mean(s2_y)
    s_beta = np.sqrt(s2_b / samples / experiments)
    stat_t = [abs(beta[i]) / s_beta for i in range(factors)]

    crit_t = t_critical(confidence_prob, (samples - 1) * experiments)

    print(f"Calculated t statistics: {[round(stat_t[i], 3) for i in range(len(stat_t))]}")
    print(f"Critical t for confidence probability of {confidence_prob}: {round(crit_t, 3)}")

    significant_coeffs = len(b)
    for i in range(len(stat_t)):
        if stat_t[i] < crit_t:
            b[i] = 0
            significant_coeffs -= 1

    print(f"Significant coefficients: {significant_coeffs}")
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
        found = True
