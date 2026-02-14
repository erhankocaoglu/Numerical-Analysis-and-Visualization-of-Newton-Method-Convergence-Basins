#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MAIN PART:
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import math
import cmath
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

tolerance_real = 1e-8    #   Sets convergence tolerance for real Newton iteratons
tolerance_complex = 1e-6  #Sets convergence tolerance for complex Newton iterations
maximum_iteration = 50  #Defines maximum number of Newton iterations
root_tolerance_real = 1e-4  # Defines acceptance radius around real roots
root_tolerance_complex = 1e-3  #Defines acceptance radius around complex roots

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Configuriing visual theme and style
# A dark visual theme was configured by defining background, grid, text, and basin colors for all plots.
# Matplotlib rcParams were updated in bulk to apply consistent styling such as facecolors and grid properties.
# A discrete colormap and boundary normalizer were created so each basin class maps to a fixed, predefined color.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

theme_name = "dark"

# Applies my chosen colors to customize graphics
if theme_name == "dark":
    background_color = "#0b0f14"
    grid_color = "#334155"
    title_color = "#f8fafc"
    foreground_color = "#e5e7eb"  # text / markers on dark background
    color_for_divergence = "#64748b"
    color_for_root_index_0 = "#7dd3fc"
    color_for_root_index_1 = "#fbbf24"
    color_for_root_index_2 = "#34d399"
    color_for_root_index_3 = "#e879f9"
else:  #Handle any non-dark theme in this fallback branch, this will not work
  foreground_color = "black"
  pass  #Leave non-dark theme unconfigured

# Apply the visual theme settings in bulk: iterate over a dictionary of rcParams to push background colors, grid colors, and spacing
for k, v in {
    "figure.facecolor": background_color,
    "axes.facecolor": background_color,
    "axes.edgecolor": grid_color,
    "grid.color": grid_color,
    "axes.titlepad": 10,
}.items():
    plt.rcParams[k] = v


# Create a list of colors representing each basin category:
discrete_color_list = [
    color_for_divergence,
    color_for_root_index_0,
    color_for_root_index_1,
    color_for_root_index_2,
    color_for_root_index_3,
]


# Builds the discrete colormap and its boundary normalizer: the colormap maps each basin class to a fixed color
colormap_for_basins = mcolors.ListedColormap(discrete_color_list, name="basins")
class_bounds_for_basins = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
normalization_for_basins = mcolors.BoundaryNorm(
    class_bounds_for_basins,
    colormap_for_basins.N
)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Buildimg real-line grid and prepare initial guesses:
# A uniform grid of real initial guesses was constructed over the interval [real_minimum, real_maximum].
# Real roots and polynomial/derivative lambdas for f1, f2, f3, and h were collected into the real_polynomials list.
# Empty containers were reserved for storing basin indices and iteration counts for each real polynomial.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
real_minimum = -6.0  # Set lower limit of real interval
real_maximum = 6.0  #Set upper limit of real interval
number_of_real_points = 1600  # Declare number of real starting points to sample
real_step_size = (real_maximum - real_minimum) / (number_of_real_points - 1)  # Compute spacing between consecutive x0 values

initial_guesses_on_real_line = [  #Create list of real initial guesses along the line
    real_minimum + i * real_step_size for i in range(number_of_real_points)  #Fill list by stepping from minimum with constant step
]


#Declare real roots for polynomial f1/2/3/
roots_of_f1 = [-1.0, 4.0]
roots_of_f2 = [-3.0, 1.0]
roots_of_f3 = [-3.0, 1.0, 4.0]
roots_of_h  = [-2.5, 0.75, 3.8]  # Declare real roots for polynomial h

real_polynomials =  [  #Collect all polynomial configs into a single list
    (  #Open configuration tuple for f1
        roots_of_f1,  # Attach root list associated with f1
        lambda x: (x - 4.0) * (x + 1.0),  # Provide lambda that computes f1(x)
        lambda x: 2.0 * x - 3.0,  # Provide lambda that computes f1'(x)
        "Real basins: f1(x) = (x - 4)(x + 1)",  #Supply title text for f1 basin plot
        "f1",  #Supply tag identifier for f1
    ),
    (  #Open configuration tuple for f2
        roots_of_f2,
        lambda x: (x - 1.0) * (x + 3.0),
        lambda x: 2.0 * x + 2.0,
        "Real basins: f2(x) = (x - 1)(x + 3)",
        "f2",
    ),
    (  #Open configuration tuple for f3
        roots_of_f3,
        lambda x: (x - 4.0) * (x - 1.0) * (x + 3.0),
        lambda x: 3.0 * x * x - 4.0 * x - 11.0,
        "Real basins: f3(x) = (x - 4)(x - 1)(x + 3)",
        "f3",
    ),
    (  #Open configuration tuple for h
        roots_of_h,  #
        lambda x: (x + 2.5) * (x - 0.75) * (x - 3.8),
        lambda x: 3.0 * x * x - 4.1 * x - 8.525,
        "Real basins: h(x) = (x + 2.5)(x - 0.75)(x - 3.8)",
        "h",
    ),
]


#Reserving containers for f1/2/3/4 basin indices and iteration counts
basin_indices_of_f1, iteration_counts_of_f1 = None, None
basin_indices_of_f2, iteration_counts_of_f2 = None, None
basin_indices_of_f3, iteration_counts_of_f3 = None, None
basin_indices_of_h,  iteration_counts_of_h  = None, None

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Execute real Newton iterations in a generic loop:
# Real Newton iterations were performed for each polynomial over all initial guesses on the real line.
# Each starting point was iterated until convergence or stopping criteria were met, then assigned to the nearest root or marked as diverged.
# Summary statistics were printed and a scatter plot was drawn with iteration counts on the y-axis and colors indicating root basins or divergence.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

for roots, f_expr, fp_expr, title_text, tag in real_polynomials:
    # Initialize containers for this polynomial
    basin_indices = []
    iteration_counts = []

    # Loop over all initial guesses on the real line
    for current_x_value in initial_guesses_on_real_line:
        # Reset convergence(its flag more accurately) and iteration counter for this starting point
        has_converged = False
        iteration_counter = 0

        # Perform Newton iterations until convergence or until meeting a break condition
        while True:
            # Evaluate function and derivative at the current point
            value_at_x = f_expr(current_x_value)
            derivative_at_x = fp_expr(current_x_value)

            # Stop iteration if derivative is too small to divide safely
            if abs(derivative_at_x) < 1e-14:
                break

            # Compute the next Newton iterate
            new_x_value = current_x_value - value_at_x / derivative_at_x

            # Measure the size of the update step
            step_size = abs(new_x_value - current_x_value)

            # Check convergence using the step size
            if step_size < tolerance_real:
                # Mark this starting point as converged
                has_converged = True

                # Store the converged value and update iteration count
                current_x_value = new_x_value
                iteration_counter += 1
                break

            # Move to the new point and increase iteration counter
            current_x_value = new_x_value
            iteration_counter += 1

            # Stop if the maximum number of iterations is reached
            if iteration_counter >= maximum_iteration:
                break

        #  non-convergent starting points:
        if not has_converged:
            # Record maximum iteration count and divergence basin
            iteration_counts.append(maximum_iteration)
            basin_indices.append(-1)
        else:
            # Record the number of iterations used for convergence
            iteration_counts.append(iteration_counter)

            # Initialize nearest root selection
            selected_root_index = 0
            minimum_distance_to_root = abs(current_x_value - roots[0])

            # Search for the closest root among all candidates
            for i in range(1, len(roots)):
                # Compute distance to the i-th root
                distance = abs(current_x_value - roots[i])

                # Update nearest root if this distance is smaller
                if distance < minimum_distance_to_root:
                    minimum_distance_to_root = distance
                    selected_root_index = i

            # Assign the label if the converged point is close enough to a root
            if minimum_distance_to_root < root_tolerance_real:
                basin_indices.append(selected_root_index)
            else:
                # Mark as divergence if no root is within tolerance
                basin_indices.append(-1)


    if tag == "f1":  #Route results into f1 containers when tag matches
        basin_indices_of_f1, iteration_counts_of_f1 = basin_indices, iteration_counts    #Assign f1 basin indices and counts
    elif tag == "f2":
        basin_indices_of_f2, iteration_counts_of_f2 = basin_indices, iteration_counts #Assign f2 basin indices and counts
    elif tag == "f3":
        basin_indices_of_f3, iteration_counts_of_f3 = basin_indices, iteration_counts  #Assign f3 basin indices and counts
    elif tag == "h":
        basin_indices_of_h,  iteration_counts_of_h  = basin_indices, iteration_counts   #Assign h basin indices and counts

    # --------------------- TEXT OUTPUT (REAL) ---------------------
    total_points_real = len(initial_guesses_on_real_line)
    diverged_count_real = sum(1 for b in basin_indices if b == -1)
    converged_count_real = total_points_real - diverged_count_real
    if converged_count_real > 0:
        avg_iterations_real = sum(
            iteration_counts[i]
            for i in range(total_points_real)
            if basin_indices[i] != -1
        ) / converged_count_real
    else:
        avg_iterations_real = float("nan")

    print(f"[REAL SUMMARY] {title_text}")
    print(f"  total starting points: {total_points_real}")
    print(f"  converged points:      {converged_count_real}")
    print(f"  diverged points:       {diverged_count_real}")
    print(f"  average iterations (converged): {avg_iterations_real:.2f}")
    # ---------------------------------------------------------------

    # ------------------------------------------------------------------
    # y-axis = iteration count, color = basin index:
    # A scatter plot was generated where x0 values were plotted against their iteration counts, with colors indicating basin membership.
    # A colorbar was added to label each basin class and divergence using the predefined theme colors.
    # Plot styling, grid, axes, and title were adjusted to match the dark theme before rendering the final visualization.
    # ------------------------------------------------------------------
    figure_object = plt.figure(figsize=(10, 3.5), facecolor=background_color)
    axis_object = plt.gca()

    x_values = np.array(initial_guesses_on_real_line)      # x0 (initial guesses)
    y_values = np.array(iteration_counts)                  # iteration count for each x0

    scatter_object = axis_object.scatter(
        x_values,
        y_values,
        c=basin_indices,
        s=8,
        marker="s",
        cmap=colormap_for_basins,
        norm=normalization_for_basins,
        linewidths=0,
        alpha=0.95,
    )

    # Add colorbar to indicate which color corresponds to which basin
    colorbar_object = plt.colorbar(
        scatter_object,
        ax=axis_object,
        ticks=[-1] + list(range(len(roots))),
        pad=0.02,
    )
    colorbar_labels = ["divergence"] + [
        f"root {i} (x={roots[i]})" for i in range(len(roots))
    ]
    colorbar_object.ax.set_yticklabels(colorbar_labels)
    colorbar_object.outline.set_edgecolor(grid_color)
    colorbar_object.ax.tick_params(axis="y", colors=foreground_color)
    for tick_label in colorbar_object.ax.get_yticklabels():
        tick_label.set_color(foreground_color)

    axis_object.grid(True, axis="both")
    axis_object.spines["top"].set_visible(False)
    axis_object.spines["right"].set_visible(False)
    axis_object.spines["left"].set_color(grid_color)
    axis_object.spines["bottom"].set_color(grid_color)
    axis_object.set_xlabel("x0 (initial guess)", color=foreground_color)
    axis_object.set_ylabel("iteration count", color=foreground_color)
    axis_object.tick_params(axis="x", colors=foreground_color)
    axis_object.tick_params(axis="y", colors=foreground_color)
    axis_object.set_ylim(-0.5, maximum_iteration + 1)
    axis_object.set_title(title_text + " — iterations vs x0", color=title_color)

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ComputING complex-plane Newton basins for g(z) = z^3 - 1
# A complex-plane grid was generated and each point was initialized for Newton iteration on g(z)=z^3−1.
# Newton’s method was applied repeatedly, and points were classified once their updates were stabilized within tolerance.
# Converged points were assigned to the nearest root basin, while non-converging points were marked with the maximum iteration count.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

number_of_complex_points_per_axis = 300  #SetS number of grid samples per complex axis
real_axis_minimum = -2.0  #Define minimum real coordinate for complex plane
real_axis_maximum = 2.0  # Define maximum real coordinate for complex plane
imaginary_axis_minimum = -2.0  # DefinES minimum imaginary coordinate for complex plane
imaginary_axis_maximum = 2.0  #Define maximum imaginary coordinate for complex plane

real_axis_step = (real_axis_maximum - real_axis_minimum) / (  #Compute spacing between real grid samples
    number_of_complex_points_per_axis - 1  #Subtract one so endpoints are included properly
)
imaginary_axis_step = (imaginary_axis_maximum - imaginary_axis_minimum) / (  #Compute spacing between imaginary grid samples
    number_of_complex_points_per_axis - 1  # Subtract one for symmetric sampling
)

real_axis_values = np.linspace(  #Generate linearly spaced real coordinates
    real_axis_minimum, real_axis_maximum, number_of_complex_points_per_axis  #Pass min, max, and number of samples
)
imaginary_axis_values = np.linspace(  #Generate linearly spaced imaginary coordinates
    imaginary_axis_minimum, imaginary_axis_maximum, number_of_complex_points_per_axis  #Pass min, max, and number of samples
)
X, Y = np.meshgrid(real_axis_values, imaginary_axis_values)  # Create 2D array from real and imaginary vectors
Z = X + 1j * Y  #Combine 2 d arrya into complex-valued grid Z

angle_in_radians = 2.0 * math.pi / 3.0  # Compute angle of 120 degrees in radians
roots_of_g = [  #List the three complex cube roots of unity for g(z)
    complex(1.0, 0.0),  # Include real root at 1 + 0i
    complex(math.cos(angle_in_radians), math.sin(angle_in_radians)),  #Include root at e^(2πi/3)
    complex(math.cos(2.0 * angle_in_radians), math.sin(2.0 * angle_in_radians)),  #Include root at e^(4πi/3)
]

basin_indices_of_g = np.full(Z.shape, -1, dtype=int)  #Initialises basin index array with -1 default for all entries
iteration_counts_of_g = np.zeros(Z.shape, dtype=int)  #Initialises iteration count array with zeros for all entries

active_mask = np.ones(Z.shape, dtype=bool)  #Mark every grid point as active for Newton iteration

for iteration_counter in range(maximum_iteration):  #Loop over allowed Newton iteration steps
    value_of_g_at_z = Z**3 - 1.0  #Compute g(z) = z^3 - 1 on entire grid
    derivative_of_g_at_z = 3.0 * Z * Z  # Compute derivative g'(z) = 3 z^2 on entire grid
    safe_mask = (np.abs(derivative_of_g_at_z) >= 1e-14) & active_mask  #Identify points with safe derivative and still active
    if not np.any(safe_mask):  # Check if there are no safe active points remaining
        break  #Exit Newton loop early when nothing remains to update

    new_Z = Z.copy()  #Create copy of Z grid to hold new Newton iterates
    new_Z[safe_mask] = (  #Update only safe active entries with Newton step
        Z[safe_mask] - value_of_g_at_z[safe_mask] / derivative_of_g_at_z[safe_mask]  #Apply Newton formula element-wise
    )

    converged_mask = (np.abs(new_Z - Z) < tolerance_complex) & safe_mask  #Locates points whose update step is below complex tolerance
    iteration_counts_of_g[converged_mask] = iteration_counter + 1         #Records iteration count for newly converged points

    if np.any(converged_mask):  #Handle classification for any points that just converged
        distances_stack = np.stack(  #Stack distance-to-root arrays along a new axis
            [np.abs(new_Z - roots_of_g[i]) for i in range(3)], axis=0  #Compute absolute distances to each of 3 roots
        )
        min_indices = np.argmin(
            distances_stack, axis=0  #Select index of closest root for every grid location
        )
        min_distances = np.take_along_axis(  #Extract minimal distance values using chosen indices
            distances_stack, min_indices[None, :, :], axis=0  #Index into stacked distances array by root index map
        )[0]
        assign_mask = converged_mask & (min_distances < root_tolerance_complex)  #Restrict assignment to converged points within tolerance
        basin_indices_of_g[assign_mask] = min_indices[assign_mask]  #Write basin indices according to nearest root index
        active_mask[converged_mask] = False  #Deactivate converged points so they are not processed again

    Z = new_Z  #Replace Z grid with the new iterates for the next loop step

iteration_counts_of_g[active_mask] = maximum_iteration  #Assign max iteration count to points that never converged

# --------------------- TEXT OUTPUT (COMPLEX) ---------------------
total_points_complex = iteration_counts_of_g.size
diverged_mask_complex = (basin_indices_of_g == -1)
diverged_count_complex = np.count_nonzero(diverged_mask_complex)
converged_mask_complex = ~diverged_mask_complex
converged_count_complex = np.count_nonzero(converged_mask_complex)
if converged_count_complex > 0:
    avg_iterations_complex = iteration_counts_of_g[converged_mask_complex].mean()
else:
    avg_iterations_complex = float("nan")

print("[COMPLEX SUMMARY] g(z) = z^3 - 1")
print(f"  total grid points:               {total_points_complex}")
print(f"  converged to a root:             {converged_count_complex}")
print(f"  divergence / unclassified:       {diverged_count_complex}")
print(f"  average iterations (converged):  {avg_iterations_complex:.2f}")
# --------------------------------------------------------------------

basin_indices_of_g = basin_indices_of_g.tolist()  # Converts basin indices array to nested Python list structure
iteration_counts_of_g = iteration_counts_of_g.tolist()  #Converts iteration counts array to nested Python list structur

# Plots the complex-plane basin image for g(z): map each grid point in Z to its basin index and show it as a colored 2D field.
# Overlays the actual roots as hollow markers, then attach a labeled colorbar so the meaning of each basin color
axis_object = plt.gca()
image_object = axis_object.imshow(
    basin_indices_of_g,
    extent=[
        real_axis_minimum,
        real_axis_maximum,
        imaginary_axis_minimum,
        imaginary_axis_maximum,
    ],
    origin="lower",
    cmap=colormap_for_basins,
    norm=normalization_for_basins,
    interpolation="nearest",
)
axis_object.scatter(
    [roots_of_g[0].real, roots_of_g[1].real, roots_of_g[2].real],
    [roots_of_g[0].imag, roots_of_g[1].imag, roots_of_g[2].imag],
    facecolors="none",
    edgecolors=foreground_color,
    linewidths=2.0,
    s=110,
    marker="o",
    zorder=3,
)
axis_object.grid(False)
axis_object.spines["top"].set_visible(False)
axis_object.spines["right"].set_visible(False)
axis_object.spines["left"].set_color(grid_color)
axis_object.spines["bottom"].set_color(grid_color)
axis_object.set_xlabel("Re(z)", color=foreground_color)
axis_object.set_ylabel("Im(z)", color=foreground_color)
axis_object.tick_params(axis="x", colors=foreground_color)
axis_object.tick_params(axis="y", colors=foreground_color)
axis_object.set_title("Complex basins: g(z) = z^3 − 1", color=title_color)

colorbar_object = plt.colorbar(
    image_object, ax=axis_object, ticks=[-1, 0, 1, 2], pad=0.02
)
colorbar_object.ax.set_yticklabels(["divergence", "root 1", "root 2", "root 3"])
colorbar_object.outline.set_edgecolor(grid_color)
colorbar_object.ax.tick_params(axis="y", colors=foreground_color)
for tick_label in colorbar_object.ax.get_yticklabels():
    tick_label.set_color(foreground_color)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------------
# Plot a convergence time map for g(z) = z^3 - 1.
# Use iteration_counts_of_g as color so each pixel encodes how many Newton steps were needed.
# Highlight regions of fast convergence (low iteration count) versus slow convergence (high iteration count).
# Keep the same spatial domain as the basin plot to make visual comparison easy.
# ------------------------------------------------------------------
figure_object = plt.figure(figsize=(7.2, 7.2), facecolor=background_color)
axis_object = plt.gca()

image_object = axis_object.imshow(
    iteration_counts_of_g,  # use iteration counts as color
    extent=[
        real_axis_minimum,
        real_axis_maximum,
        imaginary_axis_minimum,
        imaginary_axis_maximum,
    ],
    origin="lower",
    cmap="magma",           # use a continuous colormap for time
    interpolation="nearest",
)

axis_object.grid(False)
axis_object.spines["top"].set_visible(False)
axis_object.spines["right"].set_visible(False)
axis_object.spines["left"].set_color(grid_color)
axis_object.spines["bottom"].set_color(grid_color)
axis_object.set_xlabel("Re(z)", color=foreground_color)
axis_object.set_ylabel("Im(z)", color=foreground_color)
axis_object.tick_params(axis="x", colors=foreground_color)
axis_object.tick_params(axis="y", colors=foreground_color)
axis_object.set_title("Convergence time map: g(z) = z^3 − 1", color=title_color)

colorbar_object = plt.colorbar(image_object, ax=axis_object, pad=0.02)
colorbar_object.ax.set_ylabel("iteration count", color=foreground_color)
colorbar_object.outline.set_edgecolor(grid_color)
colorbar_object.ax.tick_params(axis="y", colors=foreground_color)
for tick_label in colorbar_object.ax.get_yticklabels():
    tick_label.set_color(foreground_color)

plt.tight_layout()
plt.show()




#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Optional PART (Damped Newton)
# Introduces a damped variant of Newton's method, where the update step
# z ← z − λ f(z)/f′(z) is scaled by a damping factor λ ∈ (0,1).
# Demonstrates how damping modifies convergence speed and stability by
# sampling multiple λ values and evaluating their effect on a fixed
# initial condition. Produces both numerical summaries and visual
# comparisons. Generates a convergence-time map over a complex grid,
# allowing observation of slow regions, stable regions, and structural
# changes in basins under damping. Illustrates how damping smooths
# chaotic boundaries and prevents excessive jumps near singularities.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Use g(z) = z^3 − 1 and g'(z) = 3z^2 without defining separate def blocks
function_g_for_damped = lambda z: z**3 - 1.0
derivative_of_g_for_damped = lambda z: 3.0 * z * z

# ----------------- Damped Newton function (kept unchanged) -----------------
def damped_newton(function_f, derivative_of_f, initial_z_value,
                  damping_factor=0.6, tolerance=1e-8, maximum_iteration=50):
    current_z_value = initial_z_value  # Initialize current iterate with the given starting value

    iteration_counter = 0  # Initialize iteration counter and convergence flag
    has_converged = False

    # Perform damped Newton steps until convergence or maximum_iteration is reached
    while iteration_counter < maximum_iteration:

        value_at_current_z = function_f(current_z_value)        # Compute function value at the current point
        derivative_at_current_z = derivative_of_f(current_z_value)        # Computes derivative value at the current point

        # Guard against division by a very small derivative
        if abs(derivative_at_current_z) < 1e-14:
            break

        damped_step = damping_factor * value_at_current_z / derivative_at_current_z  # Generate a damped Newton step
        new_z_value = current_z_value - damped_step        # Update candidate point using the damped Newton correction

        # Check convergence using the size of the update step
        if abs(new_z_value - current_z_value) < tolerance:
            has_converged = True           # Mark this starting value as converged
            current_z_value = new_z_value   # Store the final iterate
            iteration_counter += 1
            break

        current_z_value = new_z_value       # Move to the new point for the next iteration
        iteration_counter += 1              # Increase iteration counter for the next loop pass

    return current_z_value, iteration_counter, has_converged


#--------------------------------------------------------------------
# Performs a λ-sweep experiment that evaluates how different damping
# factors influence convergence. Computes iteration counts for each λ
# and records success/failure for numerical comparison.
#--------------------------------------------------------------------

lambda_values = np.linspace(0.1, 0.9, 30)   # sampling damping factors
iteration_results = []
convergence_flags = []

initial_test_point = complex(0.7, -0.4)     # reference initial value

for lam in lambda_values:
    final_z, steps, success = damped_newton(
        function_g_for_damped,
        derivative_of_g_for_damped,
        initial_test_point,
        damping_factor=lam,
        tolerance=1e-8,
        maximum_iteration=50
    )
    iteration_results.append(steps)
    convergence_flags.append(success)

print("\n[DAMPED NEWTON λ-SWEEP RESULTS]")
for i, lam in enumerate(lambda_values):
    print(f"  λ = {lam:.2f}  |  iterations = {iteration_results[i]:2d}  |  converged = {convergence_flags[i]}")


#--------------------------------------------------------------------
# Generates a plot of λ versus iteration count. Reveals how damping
# controls step size and alters the convergence rate. Highlights the
# stability window where convergence is most reliable.
#--------------------------------------------------------------------

plt.figure(figsize=(8, 4), facecolor=background_color)
axis = plt.gca()

axis.plot(lambda_values, iteration_results, marker="o", color=color_for_root_index_1)
axis.set_xlabel("damping factor λ", color=foreground_color)
axis.set_ylabel("iteration count", color=foreground_color)
axis.set_title("Damped Newton: iteration count vs λ", color=title_color)

axis.tick_params(axis="x", colors=foreground_color)
axis.tick_params(axis="y", colors=foreground_color)

axis.grid(True, color=grid_color)

plt.tight_layout()
plt.show()


#--------------------------------------------------------------------
# Computes a 2D convergence-time map by applying the damped Newton
# update across a complex grid. Encodes iteration count as color and
# visualizes how damping reshapes convergence regions. Shows smooth
# transitions, reduced instability, and moderated fractal structures.
#--------------------------------------------------------------------

grid_points = 200
rx = np.linspace(-2, 2, grid_points)
ry = np.linspace(-2, 2, grid_points)
XX, YY = np.meshgrid(rx, ry)
ZZ = XX + 1j * YY

iteration_map = np.zeros_like(ZZ, dtype=int)

lam = 0.6  # chosen damping factor

for i in range(grid_points):
    for j in range(grid_points):
        _, steps, _ = damped_newton(
            function_g_for_damped,
            derivative_of_g_for_damped,
            ZZ[i, j],
            damping_factor=lam,
            tolerance=1e-8,
            maximum_iteration=50
        )
        iteration_map[i, j] = steps

plt.figure(figsize=(7, 7), facecolor=background_color)
axis = plt.gca()

image = axis.imshow(
    iteration_map,
    extent=[-2, 2, -2, 2],
    origin="lower",
    cmap="inferno"
)

cbar = plt.colorbar(image, ax=axis, pad=0.02)
cbar.ax.set_ylabel("iteration count", color=foreground_color)
cbar.outline.set_edgecolor(grid_color)
cbar.ax.tick_params(axis="y", colors=foreground_color)
for tick_label in cbar.ax.get_yticklabels():
    tick_label.set_color(foreground_color)

axis.set_title("Damped Newton convergence time map (λ = 0.6)", color=title_color)
axis.set_xlabel("Re(z)", color=foreground_color)
axis.set_ylabel("Im(z)", color=foreground_color)

axis.tick_params(axis="x", colors=foreground_color)
axis.tick_params(axis="y", colors=foreground_color)

axis.spines["top"].set_visible(False)
axis.spines["right"].set_visible(False)
axis.spines["left"].set_color(grid_color)
axis.spines["bottom"].set_color(grid_color)

plt.tight_layout()
plt.show()
