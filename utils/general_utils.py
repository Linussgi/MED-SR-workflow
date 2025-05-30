import pandas as pd
import sympy as sp
import numpy as np
import glob
import os
import shutil


def create_function(equation_str: str, *param_names: str) -> callable:
    """
    Converts a string representation of a function into a callable using sympy.lambdify.

    Args:
        equation_str (str): The equation string, e.g., "x + y".
        *param_names (str): Parameter names used in the equation.

    Returns:
        callable: A NumPy-compatible function that evaluates the equation.
    """
    equation_str = equation_str.replace("^", "**")

    symbols = sp.symbols(param_names)  # Create sympy symbols dynamically
    expr = sp.sympify(equation_str, locals={"exp": sp.exp, "log": sp.log})

    func = sp.lambdify(symbols, expr, "numpy")

    return func


def split_df(df: pd.DataFrame, split_frac: float, seed=100) -> tuple[pd.DataFrame]:
    """Splits a dataframe into a train section and test section for training a model"""

    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(split_frac * len(df_shuffled))
    train_df, test_df = df_shuffled.iloc[:split_idx], df_shuffled.iloc[split_idx:]

    return train_df, test_df


def find_hof_file(pattern) -> str | None:
    """
    Searches for the most recent hall_of_fame.csv file in the temporary directory.
    If multiple files are found, it selects the most recently modified one.
    
    Returns:
        str | None: The path to the most recent hall_of_fame.csv file, or None if no file is found.
    """
    files = glob.glob(pattern, recursive=True)

    if len(files) == 0:
        print("No hall_of_fame.csv file found.")
        return None
    elif len(files) > 1:
        # If multiple files are found, find the most recent one
        print(f"Multiple hall_of_fame.csv files found, selecting the most recent one.")
        most_recent_file = max(files, key=os.path.getmtime)
        print(f"Most recent hall_of_fame.csv is: {most_recent_file}")
        return most_recent_file
    
    return files[0]


def move_hof_file(dest_dir: str, hof_file: str) -> bool:
    """
    Finds the most recent hall_of_fame.csv file in the temp directory, moves it to dest_dir,
    and ensures no old file remains in the temporary location.
    
    Args:
        dest_dir (str): The destination directory to move the file to.
        
    Returns:
        bool: True if the file was successfully moved, False otherwise.
    """
    
    if not hof_file:
        print("No hall_of_fame.csv file found")
        return False
    else:
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        dest_file = os.path.join(dest_dir, "hall_of_fame.csv")

        try:
            shutil.move(hof_file, dest_file)
            print(f"Moved hall_of_fame.csv from {hof_file} to {dest_file}")

            return True
        except Exception as e:
            print(f"Error moving file: {e}")

            return False