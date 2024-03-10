import numpy as np
from equationsolver_double_pendulum import sol1, l1, l2


def get_coordinates(alpha, beta, l1, l2):  # Umwandlung Winkel -> Koordinaten
        return (l1*np.sin(alpha),
                -l1*np.cos(alpha),
                l1*np.sin(alpha)+l2*np.sin(beta),
                -l1*np.cos(alpha)-l2*np.cos(beta))


x1, y1, x2, y2 = get_coordinates(sol1.T[0], sol1.T[1], l1, l2)



