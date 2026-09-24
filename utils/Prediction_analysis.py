import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import os


def prediction_analysis(true_values, pred_values, model_name,
                        mae, mse, r2, mape, save_dir="outputs"):
    """
    Draw model prediction analysis figures: predicted‑vs‑true scatter plot,
    error distribution histogram, and error sequence plot.

    Parameters
    ----------
    true_values : list / np.ndarray
        Ground‑truth values
    pred_values : list / np.ndarray
        Model predicted values
    model_name : str
        Model name, used for figure title and output filename
    mae : float
        Mean absolute error
    mse : float
        Mean squared error
    r2 : float
        Coefficient of determination (R‑squared)
    mape : float
        Mean absolute percentage error
    save_dir : str
        Directory for saving output figures, default as "outputs"
    """
    # Convert inputs to numpy array for numerical computation compatibility
    true_values = np.array(true_values)
    pred_values = np.array(pred_values)

    # Calculate prediction errors
    errors = pred_values - true_values
    abs_errors = np.abs(errors)

    # Reset matplotlib font settings to default
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['font.size'] = 10

    # Create figure with 2‑row 2‑column subplot layout
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Subplot 1: predicted value versus true value scatter plot
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(true_values, pred_values,
                          c=errors, cmap='coolwarm', alpha=1, s=25, edgecolors='black')

    # Draw ideal reference line (y=x)
    min_val = min(min(true_values), min(pred_values))
    max_val = max(max(true_values), max(pred_values))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=1.5, label='Ideal Line (y=x)')

    # Perform linear regression and draw fitted line
    slope, intercept, r_value, p_value, std_err = stats.linregress(true_values, pred_values)
    x_fit = np.linspace(min_val, max_val, 100)
    y_fit = intercept + slope * x_fit
    ax1.plot(x_fit, y_fit, 'b--', linewidth=1.5, label="Fitted Line")

    # Compute and draw 95% confidence interval
    n = len(true_values)
    df = n - 1
    t_critical = stats.t.ppf(0.975, df)
    error_std = np.std(errors, ddof=1)
    ideal_center = x_fit
    conf_offset = t_critical * error_std
    conf_int_lower = ideal_center - conf_offset
    conf_int_upper = ideal_center + conf_offset

    ax1.plot(x_fit, conf_int_upper, 'g--', linewidth=1, label='95% Confidence Boundary')
    ax1.plot(x_fit, conf_int_lower, 'g--', linewidth=1)
    ax1.fill_between(x_fit, conf_int_lower, conf_int_upper, color='lightgreen', alpha=0.2, zorder=1)

    # Display quantitative evaluation metrics on subplot
    text_str = (
        f"{'  MAE':>4}={mae:.6f}\n"
        f"{'  MSE':>4}={mse:.6f}\n"
        f"{'     R²':>4}={r2:.6f}\n"
        f"{'MAPE':>4}={mape * 100:.2f}%"
    )
    ax1.text(0.03, 0.97, text_str,
             transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top',
             horizontalalignment='left')

    ax1.set_xlabel('True Value')
    ax1.set_ylabel('Predicted Value')
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Add colorbar for scatter error
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Error')

    # Subplot 2: prediction error distribution histogram
    ax2 = fig.add_subplot(gs[0, 1])
    counts, bins, patches = ax2.hist(errors, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    mean_error = np.mean(errors)
    std_error = np.std(errors, ddof=1)
    ci_lower = mean_error - 1.96 * std_error
    ci_upper = mean_error + 1.96 * std_error

    # Draw reference vertical lines
    ax2.axvline(x=0, color='red', linewidth=1.5, label='Zero Error Line')
    ax2.axvline(mean_error, color='blue', linewidth=1, label=f'Mean Error : {errors.mean():.4f}')
    ax2.axvline(ci_lower, color='darkgreen', linestyle='--', linewidth=1, label='95% Confidence Boundary')
    ax2.axvline(ci_upper, color='darkgreen', linestyle='--', linewidth=1)

    ax2.legend(fontsize=8, loc='upper left')
    ax2.set_xlabel('Error (Predicted - True)')
    ax2.set_ylabel('Sample Count')
    ax2.set_title('Prediction Error Distribution')

    # Subplot3: prediction error sequence plot (occupies full bottom row)
    ax3 = fig.add_subplot(gs[1, :])
    sample_num = min(4000, len(errors))
    ax3.scatter(range(sample_num), errors[:sample_num], alpha=0.6, s=8, color='purple')
    ax3.axhline(0, color='red', linestyle='-', linewidth=1.5, label='Zero Error Line')
    ax3.axhline(mae, color='orange', linestyle='--', linewidth=2, label=f'MAE: {mae:.6f}')
    ax3.axhline(-mae, color='orange', linestyle='--', linewidth=2)

    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Error (Predicted - True)')
    ax3.set_title(f'Prediction Error Sequence ( {sample_num} Samples)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Overall figure title
    fig.suptitle(f'{model_name} Model Prediction Analysis', fontsize=14, fontweight='bold')

    # Create output folder and save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}_prediction_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Prediction analysis figure saved to: {save_path}")
    plt.show()
