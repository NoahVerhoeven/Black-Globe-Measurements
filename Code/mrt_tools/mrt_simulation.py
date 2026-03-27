def grey_body_MRT_estimate(T, h, T_a, epsilon):
    sigma = 5.67 * 10 ** -8

    return (T ** 4 + h * (T - T_a) / (sigma * epsilon)) ** 0.25


def dTdt(t, T, MRT_func, args):
    h,  T_a, epsilon, constant, A_shell, A_i, h_i, constant_i = args
    sigma = 5.67 * 10 ** -8

    T_i, T_g = T[0], T[1]

    dT_i = ((A_i * h_i) / constant_i) * (T_g - T_i)
    dT_g = (A_shell * (sigma * epsilon * MRT_func(t) ** 4 - sigma * epsilon * T_g ** 4 - h(t) * (T_g - T_a))  - A_i * h_i * (T_g - T_i)) / constant

    return [dT_i, dT_g]


