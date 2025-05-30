# MED Symbolic Regression Framework

A modular framework for running and analysing symbolic regression using the [MED](https://github.com/uob-positron-imaging-centre/MED/tree/main) (Model Equation Discovery) library, itself built on PySR. This workflow acts as a high level wrapper for the MED symbolic regression process, automating it over a number of parameters such as random seed, multiple target columns or multiple datasets, with evaluation and visualisation of regression results. 

## Project Structure

```
.
├── med_fitting.py          # Entry point for batch regression execution
├── med_input_csvs/         # Your input CSV datasets
├── med_post/               # Output folder for MED results
├── utils/
│   ├── general.py          # Utility functions for data prep and equation evaluation
│   ├── plotting.py         # Heatmap plotting tools
│   └── MEDProcessor.py     # Core class for running MED regressions
└── README.md
```

## Requirements

- numpy>=1.24,<2.0
- pandas>=1.5
- matplotlib>=3.5
- sympy>=1.11
- medeq>=0.2.1

## How to Use

### 1. Prepare Input

- Place CSV files in `med_input_csvs/` which contain training data for MED symbolic regressor.
- A CSV file should contain labelled data points as rows. For example `feature1, feature2, feature3, target`. 
- The user will later specify which columns are feature/target columns, with the other columns in the CSV being ignored

### 2. Define MED Regressions

The user can define a number of parameters with which to conduct the regression. Parameters include: 

- Target column (multiple target columns can be selected, with a regression for each target)
- Random seed to split the test-train data with

```bash
python med_fitting.py
```

- Iterates through combinations of seeds, parameter sweeps, and target dimensions
- Results are saved to `med_post/`

## Module Descriptions

### `med_fitting.py`

- Loads input data
- Configures regression combinations
- Runs batch regression using `MEDProcessor`

### `utils/MEDProcessor.py`

The `MEDProcessor` class handles the data input, MED output and evalution of MED results. 

```py
import pandas as pd
from MEDProcessor import MEDProcessor

# Load your dataset into a pandas DataFrame
df = pd.read_csv("dataset.csv")

# Specify the names of the parameter columns and the target column
param_names = ["x1", "x2", "x3"]
target_name = "y"

# Initialise the MEDProcessor
processor = MEDProcessor(
    param_names=param_names,
    df=df,
    target=target_name,
    folder_save_name="med_study"  # folder where results will be saved
)

# Split the data into training and testing sets and get MED parameter config
train_df, test_df, parameters = processor.prepare_data(split_frac=0.8)

# Run symbolic regression with MED on the training data
processor.run_med_discovery(train_df, parameters)

# Test the discovered equations on the test dataset
results_df = processor.test_equations(test_df)

# Retrieve a specific equation by its complexity level
eq = processor.get_complexity_equation(complexity=5)
print(f"Equation at complexity 5: {eq}")
```

`MEDProcessor` supports an arbitrary number of parameters. As seen in the example `med_fitting.py`, the MEDProcessor class can be used to easily set up large batches of MED regressions with many different parameters.

### `utils/general.py`

- `split_df()`: Splits data into training and testing sets.
- `create_function()`: Converts symbolic strings into Python callables.
- `find_hof_file()`: Current version of MED (0.2.1) has a bbug which saves equation results (hall_of_fame.csv) to the tmp directories on macOS. This function locates the results file, and returns its path.
- `move_hof_file()`: Moves the MED equation results file to the directory containing the rest of the MED results.

## Output

Saved in `med_post/<specified_name>/`. In addition to the default MED output files, this workflow also saves a file called `med_unseen.csv` to the target directory. This csv contains information regarding the performance of each MED equation discovered in the regression. Each equation is evaluated on test data points, with the percentage error for each equation on each test data point recorded in the CSV.
