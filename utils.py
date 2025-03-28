import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
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


def plot_derivative(f, df, x=np.linspace(-5, 5, 100), t: str | None = None, n=5):
    """
    Plot the derivative of a function as vectors along the function.
    ### Parameters
    - `f` (function): function to plot
    - `df` (function): derivative of the function
    - `x` (array-like): range of x values to plot
    - `t` (str | None): title of the plot, optional
    - `n` (int): number of points to plot the derivative at
    """
    x_derivatives = np.linspace(x.min(), x.max(), n)
    r = 1
    angles = np.arctan(df(x_derivatives))
    xx = np.vstack(
        [x_derivatives - r * np.cos(angles), x_derivatives + r * np.cos(angles)]
    ).T
    yy = np.vstack(
        [f(x_derivatives) - r * np.sin(angles), f(x_derivatives) + r * np.sin(angles)]
    ).T
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=f(x), mode="lines", name="f(x)"))
    for i in range(n):
        fig.add_trace(
            go.Scatter(
                x=xx[i],
                y=yy[i],
                marker=dict(size=15, color="red"),
                mode="lines",
                line=dict(width=2),
                showlegend=i == 0,
                name="df(x)",
            )
        )
    title = "Derivative as vectors along the function" + (f" - {t}" if t else "")
    fig.update_layout(title=title, xaxis_title="x", yaxis_title="y")
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    fig.show()


def plot_gradient_1d(f, df, x, n=10):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=f(x), mode="lines", name="f(x)"))

    x_n = np.linspace(x.min(), x.max(), n)

    for i in range(n):
        x_i = x_n[i]
        x_i_next = x_i + df(x_i)
        fig.add_trace(
            go.Scatter(
                x=[x_i, x_i_next],
                y=[0, 0],
                mode="markers+lines",
                showlegend=i == 0,
                name=f"Gradient",
                marker=dict(
                    size=10,
                    color="red",
                    symbol="arrow-bar-up",
                    angleref="previous",
                ),
            )
        )

    fig.update_layout(title="Gradient in 1D", xaxis_title="x", yaxis_title="f(x)")
    fig.show()


def plot_gradient_2d(
    f,
    df,
    x1=np.linspace(-10, 10, 100),
    x2=np.linspace(-10, 10, 100),
    n=10,
    follow_surface=False,
):
    x1, x2 = np.meshgrid(x1, x2)
    z = f(x1, x2)
    fig = go.Figure(
        data=[
            go.Surface(
                z=z, x=x1, y=x2, colorscale="Viridis", opacity=0.8, showscale=False
            ),
        ]
    )

    # now we need to plot the gradient as vectors that lie in the horizontal plane (input space)
    x1_n = np.linspace(x1.min(), x1.max(), n)
    x2_n = np.linspace(x2.min(), x2.max(), n)
    x1_n, x2_n = np.meshgrid(x1_n, x2_n)

    for i in range(n):
        for j in range(n):
            x1_i = x1_n[i, j]
            x2_i = x2_n[i, j]
            g = df(x1_i, x2_i)
            if follow_surface:
                z_start = f(x1_i, x2_i)
                z_end = f(x1_i + g[0], x2_i + g[1])
                delta_z = z_end - z_start
                sizeref = 0.3
            else:
                z_start = 0
                z_end = 0
                delta_z = 0
                sizeref = 0.8

            # Scatter3d trace (unchanged)
            fig.add_trace(
                go.Scatter3d(
                    x=[x1_i, x1_i + g[0]],
                    y=[x2_i, x2_i + g[1]],
                    z=[z_start, z_end],
                    mode="lines",
                    line=dict(color="red", width=2),
                    showlegend=False,
                )
            )

            # Updated Cone trace with vertical component
            fig.add_trace(
                go.Cone(
                    x=[x1_i + g[0]],
                    y=[x2_i + g[1]],
                    z=[z_end],
                    u=[g[0]],
                    v=[g[1]],
                    w=[delta_z],  # Vertical component added here
                    showscale=False,
                    colorscale="Reds",
                    sizemode="scaled",
                    sizeref=sizeref,
                )
            )
    l = {"l": 0, "r": 0, "b": 0, "t": 0}
    fig.update_layout(go.Layout(margin=go.layout.Margin(**l)))
    fig.update_layout(
        title="Gradient and function",
        autosize=False,
        width=800,  # height=600
    )
    fig.show()


def create_2d_meshpoints(X, resolution=200):
    """
    Takes a dataset and returns a 2d meshgrid, and the dimensionally reduced dataset.

    Parameters
    ----------
    X : np.ndarray
        The feature matrix of shape (n_samples, n_features)
    resolution : int
        The resolution of the meshgrid used to plot the decision boundary

    Returns
    -------
    mesh_points : np.ndarray
        The meshgrid of shape (resolution**2, n_features)
    xx : np.ndarray
        The meshgrid of shape (resolution, resolution)
    yy : np.ndarray
        The meshgrid of shape (resolution, resolution)
    X_2d : np.ndarray
        The dimensionally reduced dataset (n_samples, 2)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    max_dims = 2
    n_features = X.shape[1]
    if n_features > max_dims:  # reduce dimensionality to 2 using PCA if needed
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=max_dims)
        X = pca.fit_transform(X_scaled)

    # Create a meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        # np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution)
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    if n_features > max_dims:
        mesh_points = pca.inverse_transform(mesh_points)
        mesh_points = scaler.inverse_transform(mesh_points)

    return mesh_points, xx, yy, X

def plot_decision_boundary_2d(X_grid, y, prob_function, xx, yy, X_2d, n_features):
    """
    Plots the decision boundary of a logistic regression model as a black line
    dividing the classes space shaded by a color map.

    Parameters
    ----------
    X_grid: np.ndarray
        The meshgrid of shape (n_samples, n_features)
    y : np.ndarray
        The target vector of shape (n_samples,)
    prob_function : callable
        Returns matrix of shape (n_samples, n_classes) with probabilities for each class and the input is an array of (n_samples, n_features)
    """
    probs = prob_function(X_grid)
    Z = np.argmax(probs, axis=1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, edgecolors="k")
    # add a black line to the decision boundary
    n_classes = len(np.unique(y))
    plt.contour(
        xx, yy, Z, levels=(np.arange(n_classes) + 0.5), colors="black", linewidths=2
    )
    if n_features > 2:
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
    else:
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
    plt.title("Decision Boundary")
    plt.show()

def plot_probability_boundary(
    probability_function,
    X,
    y,
    resolution=200,
):
    n_features = X.shape[1]
    mesh_points, xx, yy, X_2d = create_2d_meshpoints(X, resolution)

    # Get probabilities for all classes
    probs = probability_function(mesh_points)
    class_indices = list(range(probs.shape[1]))

    # Create subplots
    n_subplots = len(class_indices)
    fig, axes = plt.subplots(1, n_subplots, figsize=(6 * n_subplots, 5))
    axes = [axes] if n_subplots == 1 else axes

    fig.suptitle("Probability and decision Boundary")
    cmap = "coolwarm"

    for ax, cls_idx in zip(axes, class_indices):
        probs_class = probs[:, cls_idx].reshape(xx.shape)

        # Plot probability contours
        ax.contourf(xx, yy, probs_class, alpha=0.8, cmap=cmap)
        ax.contour(xx, yy, probs_class, levels=[0.5], colors="black", linewidths=2)

        # Scatter plot of data points
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, edgecolors="k", s=20)

        ax.set_title(f"Probability of Class {cls_idx}")
        if n_features > 2:
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
        else:
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")

        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

        fig.colorbar(mappable, ax=ax, label=f"Probability")

    plt.tight_layout()
    plt.show()

def plot_combined_probability_boundary(
    probability_function,
    X_grid,
    X_2d,
    xx,
    yy,
    X,
    y,
    resolution=200,
    alpha=0.6,
    linewidth=2,
):
    from utils import create_2d_meshpoints
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.colors import to_rgba
    
    mesh_points = X_grid

    # Get probabilities for all classes
    probs = probability_function(mesh_points)
    class_indices = list(range(probs.shape[1]))
    n_classes = len(class_indices)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    class_colors = colors[:n_classes]  # Ensure enough colors
    
    # Create colormap for data points
    scatter_cmap = ListedColormap(class_colors)
    
    # Compute max class and their probabilities for each mesh point
    max_probs = np.max(probs, axis=1)
    max_classes = np.argmax(probs, axis=1)
    
    # Reshape to grid
    max_probs_grid = max_probs.reshape(xx.shape)
    max_classes_grid = max_classes.reshape(xx.shape)
    
    # Convert class colors to RGBA (without alpha)
    class_rgba = np.array([to_rgba(color) for color in class_colors])
    
    # Create RGB array using max_classes_grid to index class colors
    rgb = class_rgba[max_classes_grid, :3]
    
    # Create alpha channel from max_probs_grid scaled by alpha parameter
    alpha_channel = max_probs_grid[..., np.newaxis] * alpha
    
    # Combine into RGBA image
    rgba_image = np.concatenate([rgb, alpha_channel], axis=2)
    
    # Determine the extent of the image
    extent = [xx.min(), xx.max(), yy.min(), yy.max()]
    
    # Plot the RGBA image
    ax.imshow(
        rgba_image,
        extent=extent,
        origin='lower',
        zorder=1,  # Ensure it's below the contours and points
    )
    
    # Plot decision boundary lines for each class
    for i, cls_idx in enumerate(class_indices):
        color = class_colors[cls_idx]
        
        # Get class probabilities and reshape
        probs_class = probs[:, cls_idx].reshape(xx.shape)
        # Plot decision boundary line
        ax.contour(
            xx, yy, probs_class,
            levels=[0.5],  # Decision boundary at 0.5 probability
            colors=[color],
            linewidths=linewidth,
            linestyles='solid',
            alpha=0.8,
            zorder=2,  # Above the image but below points
        )
    
    # Plot data points with class colors
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=y,
        cmap=scatter_cmap,
        edgecolors='k',
        s=40,
        zorder=3  # Ensure points are on top
    )
    
    # Add labels and title
    ax.set_title("Combined Decision Boundaries")
    if X.shape[1] > 2:
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
    else:
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
    
    # Create legend for decision boundaries
    legend_elements = [
        plt.Line2D([0], [0], color=color, lw=linewidth, 
                   label=f'Class {cls_idx} Boundary')
        for cls_idx, color in zip(class_indices, class_colors)
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.show()
