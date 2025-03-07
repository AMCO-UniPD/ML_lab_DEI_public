import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from ipywidgets import interact

def get_interactive_plot_pca():
    # generate 2d data as a line with noise
    np.random.seed(0)
    n_points = 10
    x = np.linspace(0, 1, n_points)
    y = 2 * x + 1 + np.random.normal(0, 0.3, n_points)

    # center the data
    x = x - np.mean(x)
    y = y - np.mean(y)

    def compute_projection(slope, x, y):
        x_proj = (x + slope * y) / (1 + slope**2)
        y_proj = slope * x_proj
        return x_proj, y_proj

    def compute_error(slope, x, y):
        x_proj, y_proj = compute_projection(slope, x, y)
        error = (x - x_proj)**2 + (y - y_proj)**2
        return np.sum(error)
    
    def compute_ssd_along_line(slope, x, y):
        x_proj, y_proj = compute_projection(slope, x, y)
        return np.sum(x_proj**2 + y_proj**2)

    def plot_error(slope):
        plt.clf()
        plt.figure(figsize=(8, 6))
        plt.plot([-1, 1], [-slope, slope], 'r', zorder=1)
        
        # Compute projections
        x_proj, y_proj = compute_projection(slope, x, y)
        
        # Compute errors
        error = compute_error(slope, x, y)
        ssd_line = compute_ssd_along_line(slope, x, y)
        
        plt.title(f"Error: {error:.2f} | SSD Along Line: {ssd_line:.2f}")
        
        # Plot distance lines
        for xi, yi, xpi, ypi in zip(x, y, x_proj, y_proj):
            plt.plot([xi, xpi], [yi, ypi], 'k--', zorder=2)  # Black dashed lines
        
        plt.scatter(x_proj, y_proj, color='green', marker='x', label="Projections", zorder=3)
        plt.scatter(x, y, label="Data points", color='lime', zorder=4)
        
        # fix the ratio of the x and y axes to equal otherwise the distances won't appear orthogonal
        plt.gca().set_aspect('equal', adjustable=None)
        
        # plot the x and y axes in light blue
        plt.axhline(0, color='dodgerblue', zorder=1)
        plt.axvline(0, color='dodgerblue', zorder=1)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.legend()
        plt.show()

    # create an interactive widget
    return interact(plot_error, slope=widgets.FloatSlider(value=0.2, min=-5, max=5, step=0.1))