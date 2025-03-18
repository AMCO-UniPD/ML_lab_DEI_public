import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import plotly.express as px
import plotly.graph_objects as go
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
        error = (x - x_proj) ** 2 + (y - y_proj) ** 2
        return np.sum(error)

    def compute_ssd_along_line(slope, x, y):
        x_proj, y_proj = compute_projection(slope, x, y)
        return np.sum(x_proj**2 + y_proj**2)

    def plot_error(slope):
        plt.clf()
        plt.figure(figsize=(8, 6))
        plt.plot([-1, 1], [-slope, slope], "r", zorder=1)

        # Compute projections
        x_proj, y_proj = compute_projection(slope, x, y)

        # Compute errors
        error = compute_error(slope, x, y)
        ssd_line = compute_ssd_along_line(slope, x, y)

        plt.title(f"Error: {error:.2f} | SSD Along Line: {ssd_line:.2f}")

        # Plot distance lines
        for xi, yi, xpi, ypi in zip(x, y, x_proj, y_proj):
            plt.plot([xi, xpi], [yi, ypi], "k--", zorder=2)  # Black dashed lines

        plt.scatter(
            x_proj, y_proj, color="green", marker="x", label="Projections", zorder=3
        )
        plt.scatter(x, y, label="Data points", color="lime", zorder=4)

        # fix the ratio of the x and y axes to equal otherwise the distances won't appear orthogonal
        plt.gca().set_aspect("equal", adjustable=None)

        # plot the x and y axes in light blue
        plt.axhline(0, color="dodgerblue", zorder=1)
        plt.axvline(0, color="dodgerblue", zorder=1)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.legend()
        plt.show()

    # create an interactive widget
    return interact(
        plot_error, slope=widgets.FloatSlider(value=0.2, min=-5, max=5, step=0.1)
    )


def plot_mse(x, y, *y_pred):
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    n = len(y_pred)
    _, axs = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    for ax, y_p in zip(axs, y_pred):
        ax.plot(x, y_p, "r")
        for i in range(len(x)):
            ax.plot([x[i], x[i]], [y[i], y_p[i]], "k--")
        ax.scatter(x, y)
        ax.set_title(f"MSE: {mse(y, y_p):.2f}")
    plt.show()


def plot_multiple_lin_reg(df, target_col, beta):
    """
    Plots a 3D scatter plot of the data and the regression plane
    """
    l = {"l": 0, "r": 0, "b": 0, "t": 0}
    l = go.Layout(margin=go.layout.Margin(**l))
    feature_cols = df.columns.difference([target_col])

    # Create a grid of feature values to plot the regression plane
    xrange = [np.linspace(df[col].min(), df[col].max(), 10) for col in feature_cols]
    grid = np.meshgrid(*xrange)
    # stack the grid so that we can multiply it by beta
    X = np.vstack([g.flatten() for g in grid]).T
    # add a column of ones for the bias before multiplying by beta
    ones = np.ones((len(X), 1))
    y_pred = np.hstack([ones, X]) @ beta
    # reshape the prediction to the shape of the grid
    y_pred = y_pred.reshape(grid[0].shape)
    fig = px.scatter_3d(df, x=feature_cols[0], y=feature_cols[1], z=target_col)
    fig.add_trace(go.Surface(x=xrange[0], y=xrange[1], z=y_pred, showscale=False))
    fig.update_layout(l)
    fig.update_scenes(aspectratio=dict(x=2, y=2, z=0.7))
    fig.show()


def plt_circle():
    circle = Circle((0, 0), 1, color="r", fill=False)
    square = Rectangle((-1, -1), 2, 2, color="b", fill=False)
    fig, ax = plt.subplots()
    ax.add_artist(circle)
    ax.add_artist(square)
    ax.set_aspect("equal")
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(-1.1, 1.1)
    return ax
