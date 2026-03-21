def grey_body_MRT_estimate(T, h, T_a=288, epsilon=0.95):
    sigma = 5.67 * 10 ** -8

    return (T ** 4 + h * (T - T_a) / (sigma * epsilon)) ** 0.25


def dTdt(t, T, MRT_func, h,  T_a=288, epsilon=0.95, constant=1, A=1, A_i=1, h_i=1, constant_i=1):
    sigma = 5.67 * 10 ** -8

    T_i, T_g = T[0], T[1]

    dT_i = ((A_i * h_i) / constant_i) * (T_g - T_i)
    dT_g = (A * (sigma * epsilon * MRT_func(t) ** 4 - sigma * epsilon * T_g ** 4 - h * (T_g - T_a))  - A_i * h_i * (T_g - T_i)) / constant

    return [dT_i, dT_g]