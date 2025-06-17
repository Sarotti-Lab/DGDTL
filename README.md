Data and Software Availability Statement
================
Pérez José A, Zanardi María M, and Sarotti Ariel M.

- [Description](#description)
- [Data Availability](#data-availability)
- [Software Availability and
  Licensing](#software-availability-and-licensing)
- [Step-by-Step Reproduction
  Workflow](#step-by-step-reproduction-workflow)
  - [**1. Repository Contents and
    Dependencies**](#1-repository-contents-and-dependencies)
  - [2. Execution of Computational
    Levels](#2-execution-of-computational-levels)
- [Contact](#contact)
- [How to Cite](#how-to-cite)

## Description

This document provides the necessary data, software, and workflow
information to reproduce the results presented in the manuscript:
**“Low-Cost, High-Accuracy Reactivity Modeling: Integrating Genetic
Algorithms and Machine Learning with Multilevel DFT calculations”**. The
contents herein are provided in accordance with established guidelines
for data and method sharing in scientific publishing. All
sub-directories are self-contained and ready for immediate execution.
They include the necessary Python scripts, input files to run the
workflows, and the key output files generated from our analyses for
verification.

## Data Availability

All datasets required to reproduce the essential findings of this study
are provided in a machine-readable format. These datasets, including
training matrices, computational cost vectors, and key input/output
files, are publicly available in the **Mendeley Data** repository under
the DOI: **10.17632/fd7y24vfdm.1**.

## Software Availability and Licensing

The complete source code, implemented in Python 3.8+, is provided to
ensure full reproducibility of the novel methods presented.

- **License**: The code is freely available and distributed under the
  **MIT License**. For full details, please consult the `LICENSE` file
  provided within the data repository.
- **Accessibility**: The source code is provided together with the
  datasets in the Mendeley Data repository. Please refer to the DOI
  mentioned in the ‘Data Availability’ section to access all files.

------------------------------------------------------------------------

## Step-by-Step Reproduction Workflow

This section details the complete workflow, repository contents,
dependencies, and execution steps for each of the three computational
levels.

### **1. Repository Contents and Dependencies**

This repository is structured to be both demonstrative of our key
findings and fundamentally reproducible. Our approach is twofold:

1.  **“Ready-to-run” Examples**: We provide complete, executable
    workflows for each key methodological stage to demonstrate their
    direct application and effectiveness.
2.  **Foundations for Reconstruction**: For more complex variations of
    the methods, we provide foundational code and templates. For
    example, within **Level 2**, the advanced `GA2`, `GA3`, and `GA4`
    models are supplied as templates for users to reconstruct, adapt,
    and build upon. A similar logic applies to other levels, where the
    provided code serves as a robust starting point for deeper
    exploration, ensuring that all our methods can be reproduced from
    their foundations.

#### **File Organization**

    .
    ├── 01_test_720LoT/                     # Level 1: Full 24x720 matrix & GA1-ML calibration model
    │   ├── mtx_train_720LoT.csv            # Training matrix (24 rows, 720 columns)
    │   ├── vec_cost_720LoT.csv             # Computational cost vector (720 entries)
    │   │   ├── GA1/                        # GA1-ML Model
    │   │   ├── 01_GA1_ML.py                # GA1-ML script
    │   │   ├── 02_GA1_ML_LOO.py            # GA1-ML Leave-One-Out Cross validation (LOOCV) script
    │   │   ├── 03_GA_ML_MLR_opt_LOOfold.py # GA1-ML with Multiple Linear Regression (MLR) fitting & LOOCV-MLR script
    │   │   ├── best_solution_data.csv      # Output from GA1-ML: renamed to X_train.csv for LOOCV-MLR script
    │   │   ├── best_solution_summary.txt   # Output from GA1-ML summary and fitness report
    │   │   ├── Report_Validation_LOO.csv   # Output of LOOCV results for GA1-ML, also include row by row files
    │   └───└── Report_Validation_LOO.csv   # Output of LOOCV results for GA1-ML with MLR, include row by row files
    ├── 02_test_576LoT/                     # Level 2: Reduced 24x576 matrix & GA_ML models
    │   ├── mtx_train_576LoT.csv            # Reduced training matrix (24 rows, 576 columns)
    │   ├── vec_cost_576LoT.csv             # Computational cost vector (576 entries)
    │   ├── GA1/                            # GA1-ML Model (identical to Level 1 workflow)
    │   │   ├── 01_GA1_ML.py                # GA1-ML script adapted for a new matrix (24x576)
    │   │   ├── 02_GA1_ML_LOO.py            # GA1-ML LOOCV script
    │   │   ├── 03_GA_ML_MLR_opt_LOOfold.py # MLR & LOOCV script
    │   │   └── output files                # Similar to csv and txt files Level 1 workflow
    │   ├── GA5/                            # GA5-ML Model (workflow identical to GA1)
    │   │   └── (same three scripts and output files)
    │   └── GA2_GA3_GA4_templates/          # Template scripts for GA2–GA4 (advanced users)
    │       ├── 01_GAj_ML.py                # j=2, fitness function with bayesian optimization, j=3 GA and MLR        │       │                                 simultaneously and j=4 GA, MLR and bayesian optimization together
    │       ├── 02_GAj_LOO_.py              # Template scripts for LOO validation
    │       └── 03_GA_ML_MLR_opt_LOOfold.py # MLR & LOOCV script
    │       └── output files                # Similar to csv and txt files Level 1 workflow
    ├── 03_DGDTL/                           # Level 3: DGDTL implementation & evaluation
    │   ├── 01_DGDTL_part_I.py              # Stage I: generates candidate solutions
    │   ├── 02_DGDTL_part_II.py             # Stage II: selects best solution via validation
    │   ├── X_train_4LoTs.csv               # Training matrix for Stage I with 4 LoTs
    │   ├── X_vldt_4LoTs.csv                # Validation matrix for Stage II with 4 LoTs
    │   ├── Solutions_DGDTL_stage_I.csv     # Output from Stage I: candidate solutions
    │   ├── Summary_DGDTL_Etapa1.txt        # Stage I summary report
    │   ├── Results_DGDTL_Stage_II.csv      # Output from Stage II: validation evaluation
    │   └── Summary_DGDTL_Stage_II.txt      # Output Stage II summary report
    ├── LICENSE                             # MIT License file
    └── README.Rmd                          # This R Markdown source file

#### **Software Dependencies**

This project requires **Python 3.8+**. The necessary Python packages
must be installed manually.

You can install the main dependencies via `pip` by running the following
command in your terminal:

``` bash
pip install numpy pandas scikit-learn deap scipy
```

A complete list of the required packages is provided below:

- `numpy`
- `pandas`
- `scikit-learn`
- `deap`
- `scipy`
- `tkinter` (This package is typically included with standard Python
  installations and is not installed via `pip`. It is used for
  folder-selection dialogs.)

### 2. Execution of Computational Levels

A comprehensive description of the methods, including workflows and
associated parameters, is provided below.

The execution workflow can be carried out in two ways:

1.  **Via Terminal**: The commands provided are for a Linux-based
    terminal (e.g., Ubuntu) but can be readily adapted for macOS or
    Windows.
2.  **Via an IDE**: Alternatively, all Python scripts (`.py`) can be run
    directly from any Integrated Development Environment that supports
    Python (e.g., VS Code, PyCharm, Spyder) on any operating system
    (Linux, macOS, or Windows).

**Note for terminal users on Windows**: The `cp` command used in the
instructions is for Linux/macOS. The equivalent command in the Windows
Command Prompt is `copy`.

#### **Level 1: GA-ML Calibration & MLR (`01_test_720LoT`)**

This level calibrates the initial hybrid framework which integrates a
genetic algorithm and machine learning (GA-ML) using the full `24x720`
matrix and subsequently fits a MLR model. The outputs from this workflow
directly reproduce the data presented in **Figure 4b** of the main
manuscript.

1.  **Navigate to Directory**: `cd 01_test_720LoT/`
2.  **Run GA1 Calibration**: Executes the GA with a population of 200
    over 5000 generations. A folder-selection dialog will appear to
    choose where to save outputs.
    - **Command**: `python 01_GA1_ML.py`
    - **Inputs**: `mtx_train_720LoT.csv`, `vec_cost_720LoT.csv`.
    - **Outputs**: `best_solution_data.csv`,
      `best_solution_summary.txt`.
3.  **Run Leave-One-Out (LOO) Validation**:
    - **Command**: `python 02_GA1_ML_LOO.py`
    - **Inputs**: `mtx_train_720LoT.csv`, `vec_cost_720LoT.csv`.
    - **Outputs**: `Report_Validation_LOO.csv` and summary/data files
      for 24 folds.
4.  **Run Multiple Linear Regression**:
    - **Prepare Input**: Rename the GA output:
      `cp best_solution_data.csv X_train.csv`.
    - **Command**: `python 03_GA_ML_MLR_opt_LOOfold.py`
    - **Outputs**: `Results_Full_Dataset.csv` and summary/prediction
      files for 24 folds.

#### **Level 2: Reduced-Matrix GA_ML Models (`02_test_576LoT`)**

At this level, five GA-ML-based models (GA1–GA5) are proposed to
optimize the selection of DFT LoTs from the reduced 24x576 matrix. It is
important to note that each of these models required specific
adjustments to its fitness function parameters to achieve optimal
performance.

The workflows for the GA1 and GA5 models (Figures 5a and 5d of the main
manuscript) are similar to those in Level 1. For example, the GA1 model
at this level used parameters determined through an interactive
optimization process, as detailed in the main manuscript and its
Supporting Information. The more advanced models (GA2, GA3, and GA4) use
specialized templates with their own unique, internally optimized
fitness functions.

##### **GA2: Adaptive Fitness Optimization**

GA2 incorporates a dynamic fitness function and Bayesian parameter
optimization to improve performance without manual tuning.

- **Key Features**:
  - **Dynamic Penalization**: Uses three error thresholds (`sh`, `sm`,
    `sl`) to scale penalties.
  - **Bayesian Optimization**: Tunes fitness parameters (`w1`, `w2`,
    etc.) automatically with `skopt.gp_minimize`.
  - **Conditional Cost Penalization**: The cost factor `f_cost` is
    adapted based on a computational cost threshold.

Below is the core logic for the fitness function in GA2:

``` python
# Optimized fitness function for GA2
# ===================================================================
# Fitness Function
# ===================================================================
def evalError(
    individual: list,
    error_mtx: np.ndarray,
    comp_cost: np.ndarray,
    w1: float,
    w2: float,
    penalty_factor: float,
    f_cost: float,
    high_penalty_factor: float,
    medium_penalty_factor: float,
    low_penalty_factor: float,
    sh: float,
    sm: float,
    sl: float
) -> float:
    """
    Calculates the fitness value for a given individual.

    This function minimizes the average error, adjusted with penalties for
    computational cost, number of features, and error outliers.

    Parameters
    ----------
    individual : list
        Binary vector representing the selected features.
    error_mtx : np.ndarray
        Matrix of errors for each sample and feature.
    comp_cost : np.ndarray
        Matrix of computational costs for each sample and feature.
    w1, w2 : float
        Weights for the error and penalty terms, respectively.
    penalty_factor : float
        Base penalty for the number of active features.
    f_cost : float
        Factor for the computational cost penalty.
    high_penalty_factor, medium_penalty_factor, low_penalty_factor : float
        Scaling factors for the tiered error outlier penalties.
    sh, sm, sl : float
        Thresholds for identifying high, medium, and low error outliers.

    Returns
    -------
    float
        The final fitness value (negative, as it's meant to be maximized).
    """
    conditions = np.array(individual, dtype=bool)
    sys_abs_errors = np.abs(error_mtx[:, conditions].sum(axis=1))
    cost = np.abs(comp_cost[:, conditions].sum(axis=1))

    f_cost = f_cost if cost > 1.5 else 0.02

    total_error = sys_abs_errors.sum()
    num_active_features = conditions.sum()
    total_avg_error = total_error / (num_active_features * len(error_mtx))

    high_penalty = np.sum(np.where(sys_abs_errors > sh * num_active_features, sys_abs_errors / num_active_features, 0)) * high_penalty_factor
    medium_penalty = np.sum(np.where(sys_abs_errors > sm * num_active_features, sys_abs_errors / num_active_features, 0)) * medium_penalty_factor
    low_penalty = np.sum(np.where(sys_abs_errors > sl * num_active_features, sys_abs_errors / num_active_features, 0)) * low_penalty_factor

    penalization = penalty_factor * num_active_features if num_active_features > 6 else (9 if num_active_features < 3 else 0)
    final_penalization = penalization + high_penalty + medium_penalty + low_penalty

    return -(total_avg_error * w1 + final_penalization * w2 + cost * f_cost)

# ===================================================================
# Bayesian Optimization 
# ===================================================================
def optimize_fitness_params(
    error_mtx: np.ndarray,
    comp_cost: np.ndarray,
    parms_ga: dict
) -> list:
    """
    Optimizes the parameters of the `evalError` fitness function
    using Bayesian optimization.

    Parameters
    ----------
    error_mtx : np.ndarray
        Matrix of errors for each sample and feature.
    comp_cost : np.ndarray
        Matrix of computational costs for each sample and feature.
    parms_ga : dict
        Dictionary of fixed genetic algorithm settings (e.g., pop size).

    Returns
    -------
    list
        A list containing the optimal set of hyperparameters found.
    """
    def fitness_for_optimization(params):
        w1, w2, penalty_factor, f_cost, high_penalty_factor, medium_penalty_factor, low_penalty_factor, sh, sm, sl = params
        _, log, hof = run_GA(error_mtx, comp_cost, 
                             [w1, w2, penalty_factor, f_cost, high_penalty_factor, medium_penalty_factor,
                              low_penalty_factor, sh, sm, sl], 
                             parms_ga)
        best_fitness = hof[0].fitness.values[0]
        return -best_fitness  

    search_space = [
        Real(0.01, 1.5, name="w1"),
        Real(0.01, 1.5, name="w2"),
        Real(0.001, 0.1, name="penalty_factor"),
        Real(0.02, 2.0, name="f_cost"),
        Real(0.1, 1.5, name="high_penalty_factor"),
        Real(0.05, 0.5, name="medium_penalty_factor"),
        Real(0.01, 0.3, name="low_penalty_factor"),
        Real(1.5, 2.5, name="sh"),
        Real(0.5, 1.0, name="sm"),
        Real(0.1, 0.5, name="sl")
    ]

    result = gp_minimize(fitness_for_optimization, search_space, n_calls=10, random_state=0)
    return result.x  # Return the optimal parameters
```

##### **GA3: Simultaneous Feature & Coefficient Selection**

GA3 integrates MLR directly into the GA’s fitness evaluation,
simultaneously optimizing feature selection and their regression
coefficients.

- **Key Features**:
  - **Constrained Regression**: Optimizes coefficients (β) so that their
    sum is 1 ($\sum\beta_{i} = 1$).
  - **No Intercept**: The model assumes a zero intercept.
  - **Adaptive Penalization**: Maintains the dynamic penalty scheme from
    GA2.

The core of GA3’s fitness function involves solving a constrained
regression for each individual:

``` python
# GA3: Fitness function with embedded Multiple Linear Regression
# ===================================================================
# Fitness Function
# ===================================================================
def evalError(
    individual: list,
    error_mtx: np.ndarray,
    comp_cost: np.ndarray,
    parms_eval: tuple
) -> tuple:
    """
    Calculates fitness by simultaneously selecting features and optimizing
    linear regression coefficients.

    This function evaluates an individual by building a constrained linear
    model (no intercept, coefficients sum to 1) with the selected features
    and then calculating the error based on that model's predictions.

    Parameters
    ----------
    individual : list
        Binary vector representing the selected features.
    error_mtx : np.ndarray
        Matrix of errors for each sample and feature.
    comp_cost : np.ndarray
        Matrix of computational costs for each sample and feature.
    parms_eval : tuple
        A tuple containing all hyperparameters for the fitness calculation
        (w1, w2, penalty_factor, f_cost, etc.).

    Returns
    -------
    tuple
        A single-element tuple containing the final fitness value, formatted
        for compatibility with the DEAP library.
    """
    # Unpack parameters
    w1, w2, penalty_factor, f_cost, high_pf, med_pf, low_pf, sh, sm, sl = parms_eval

    conditions = np.array(individual, dtype=bool)

    # --- Constrained Linear Regression ---
    X = error_mtx[:, conditions]
    y = np.zeros(X.shape[0])  

    # Define the objective function (Mean Squared Error) for the MLR
    def objective(beta, X, y):
        predictions = X @ beta
        return np.mean((y - predictions) ** 2)

    # Define the constraint: sum of coefficients must be 1
    def constraint(beta):
        return np.sum(beta) - 1

    # Set initial guess and run the optimization to find coefficients
    beta_initial = np.ones(X.shape[1]) / X.shape[1]
    constraints = {'type': 'eq', 'fun': constraint}
    result = minimize(objective, beta_initial, args=(X, y), constraints=constraints)
    coeffs = result.x

    # --- Fitness Calculation ---
    # Calculate error using the newly optimized regression coefficients
    sys_abs_errors = np.abs(np.dot(error_mtx[:, conditions], coeffs))

    # Calculate cost and apply adaptive factor
    cost = np.abs(comp_cost[:, conditions].sum(axis=1))
    f_cost = f_cost if cost > 1.5 else 0.02

    # Calculate average error
    num_active_features = conditions.sum()
    total_avg_error = sys_abs_errors.sum() / (num_active_features * len(error_mtx))

    # Calculate tiered penalties for outliers
    high_penalty = np.sum(np.where(sys_abs_errors > sh * num_active_features, sys_abs_errors / num_active_features, 0)) * high_pf
    medium_penalty = np.sum(np.where(sys_abs_errors > sm * num_active_features, sys_abs_errors / num_active_features, 0)) * med_pf
    low_penalty = np.sum(np.where(sys_abs_errors > sl * num_active_features, sys_abs_errors / num_active_features, 0)) * low_pf

    # Calculate penalty for feature count
    if num_active_features > 10:
        penalization = num_active_features
    else:
        penalization = penalty_factor * num_active_features if num_active_features > 6 else (9 if num_active_features < 3 else 0)

    final_penalization = penalization + high_penalty + medium_penalty + low_penalty

    return -(total_avg_error * w1 + final_penalization * w2 + cost * f_cost),
```

##### **GA4: Combined Model**

GA4 combines the Bayesian optimization of GA2 with the integrated MLR of
GA3, making it the most computationally expensive model.

#### **Level 3: Dynamic Generalization-Driven Transfer Learning (`03_DGDTL`)**

This level implements a two-stage DGDTL protocol to enhance
generalization and avoid overfitting. The optimal 4-LoTs solution
derived from this approach corresponds to the results presented in
Figure 8c of the main manuscript.

1.  **Stage I: Generation of Local Solutions**: This stage explores the
    solution space to generate a diverse set of local optima.
    - **Command**: `01_DGDTL_part_I.py`.
    - **Input**: `X_train_4LoTs.csv`.
    - **Outputs**: `Solutions_DGDTL_stage_I.csv`,
      `Summary_DGDTL_Etapa1.txt`.
2.  **Stage II: Selection via Validation Set**: This stage selects the
    best solution from Stage I by evaluating candidates on a separate
    validation set.
    - **Command**: `02_DGDTL_part_II.py`.
    - **Inputs**: `X_train_4LoTs.csv`, `X_vldt_4LoTs.csv`,
      `Solutions_DGDTL_stage_I.csv`.
    - **Outputs**: `Results_DGDTL_Stage_II.csv`,
      `Summary_DGDTL_Stage_II.txt`.

**Execution Command**: These stages must be run sequentially.

------------------------------------------------------------------------

## Contact

For any questions regarding data, software, or reproducibility, please
contact the corresponding author.

## How to Cite

This software was developed for a manuscript currently under peer
review. If you use the DGDTL method or this code in your research, we
kindly request that you cite the resulting publication.

The citation details below will be completed and updated upon
publication.

**Anticipated BibTeX Entry:**

``` bibtex
@article{Perez_YYYY_DGDTL,
  author  = {Pérez, José A. and Zanardi, María M. and Sarotti, Ariel M.},
  title   = {Low-Cost, High-Accuracy Reactivity Modeling: Integrating Genetic Algorithms and Machine Learning with Multilevel DFT calculations},
  journal = {[Scientific Journal - To be updated upon publication]},
  year    = {YYYY},
  volume  = {VV},
  pages   = {PPPP-PPPP},
  doi     = {[To be updated upon publication]}
}
```
