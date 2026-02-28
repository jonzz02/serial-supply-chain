import numpy as np


def env_step(rng, I1, I2, B1, B2, U1p, U2p, s1, s2, lam, h1, h2, p_bo, alpha):
    I1 += U1p
    I2 += U2p
    O1, O2 = max(0, s1 - (I1 - B1)), max(0, s2 - (I2 - B2))
    ship = min(I2, B2 + O1)
    I2 -= ship
    B2 = B2 + O1 - ship
    U1, U2 = ship, O2
    D = int(rng.poisson(lam))
    sales = min(I1, B1 + D)
    I1 -= sales
    B1 = B1 + D - sales
    H1 = (h1 + h2) * I1 + alpha * p_bo * B1
    H2 = h2 * (I2 + U1) + (1.0 - alpha) * p_bo * B1
    return I1, I2, B1, B2, U1, U2, D, float(H1), float(H2)
