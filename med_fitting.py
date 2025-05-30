from utils.MEDProcessor import MEDProcessor
import pandas as pd
import itertools
import time

from utils.general_utils import find_hof_file, move_hof_file


def run_med_processor(
        in_df_path: str, 
        folder_save_path: str, 
        param_list: list[str],
        target: str, 
        *, 
        split_seed: int=100, 
        tt_split: int=0.7
    ) -> None:
    """
    Run the MED Processor for a study.
    
    Args:
    - in_df_path (str): The path to the CSV containing the labelled data.
    - save_folder (str): The path of the folder that results will be saved to.
    - param_list (list[str]): A list of the names of the parameters as called in the in_df_path CSV.
    - target (str): The name of the target label to predict as called in in_df_path CSV.

    Keyword Args:
    - split_seed (int): Random seed to select the test-train data with (default is 100).
    - tt_split (float): Proportion of data to use as training data (default is 0.7).
    """
    print("=" * 75)
    print(f"Running MED on - Study: {param_list}, Target: {target}, Seed: {split_seed}")
    print(f"MED Study folder Created: {folder_save_path}")

    df = pd.read_csv(in_df_path)
    
    med_study = MEDProcessor(param_list, df, target, folder_save_path)
    train_df, test_df, parameters = med_study.prepare_data(tt_split, split_seed)
    med_study.run_med_discovery(train_df, parameters)

    # MED saved discovered equations to /tmp on macos. Issue with the version of PySR used in MED.
    # For now, this fix locates the hof file in /tmp and moves it to the correct directory 
    hof_pattern = "/var/folders/lp/*/T**/**/hall_of_fame.csv"
    hof_path = find_hof_file(hof_pattern)
    move_hof_file(folder_save_path, hof_path)
    time.sleep(0.1)

    # Test MED equations on unseen data
    df_results = med_study.test_equations(test_df)
    test_results_path = f"{folder_save_path}/med_unseen.csv"
    df_results.to_csv(test_results_path)

    print(f"Med test results saved to '{test_results_path}'")


# User defined MED regresssion parameters

# Random train-test split seeds
SEEDS = [88, 92, 50, 57, 73, 77, 75, 93, 98, 96]

# Parameter names that you wish to investigate in batches
STUDIES_PARAMS = ["param1-param2-param3", "param1-param3", "param2-param3"]
CSV_NAME = "data.csv"

med_regression_params = list(itertools.product(SEEDS, STUDIES_PARAMS))
total_regressions = len(med_regression_params)

# START INFO TABLE ================================================
TABLE_WIDTH = 60
BORDER = "+" + "-" * (TABLE_WIDTH - 2) + "+"

header = "MED REGRESSION BATCH"
header_line = "| " + header.center(TABLE_WIDTH - 4) + " |"

info_lines = [
    f"Total regressions: {total_regressions}",
    f"SEEDS: {', '.join(map(str, SEEDS))}",
    f"STUDIES_PARAMS: {', '.join(STUDIES_PARAMS)}",
    f"CSV: {CSV_NAME}"
]

formatted_info_lines = [
    "| " + line.ljust(TABLE_WIDTH - 4) + " |" for line in info_lines
]

print(BORDER)
print(header_line)
print(BORDER)
for line in formatted_info_lines:
    print(line)
print(BORDER)
# END INFO TABLE ================================================

# Iterate through MED regression parameters and perform regressions
for index, (seed, study_params) in enumerate(med_regression_params):
    med_in_path = f"med_input_csvs/{CSV_NAME}"
    med_out_path = f"med_post/med_{seed}"

    sweep_parameter_list = study_params.split("-")
    target = "target_col"

    run_med_processor(med_in_path, med_out_path, sweep_parameter_list, target, split_seed=seed)
    print(f"Finished regression {index + 1} out of {total_regressions} ({100 * index / total_regressions:.2f}%)")

    break
