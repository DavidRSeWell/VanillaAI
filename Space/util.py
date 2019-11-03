import numpy as np

def get_circle(radius, num_points=100):

    x_points = np.linspace(-radius, radius, num_points)
    circle_points_y = []
    circle_points_x = []

    for x_i in x_points:
        y_i = np.sqrt(radius ** 2 - x_i ** 2)
        circle_points_x.append(x_i)
        circle_points_x.append(x_i)
        circle_points_y.append(y_i)
        circle_points_y.append(-1.0 * y_i)

    return circle_points_y, circle_points_x
