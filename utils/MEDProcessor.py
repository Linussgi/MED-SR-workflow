import numpy as np
import pandas as pd
import medeq
from .general_utils import create_function, split_df


class MEDProcessor:
    def __init__(self, param_names: list[str], df: pd.DataFrame, target: str, folder_save_name="med_study"):
        """
        Initialises the MEDProcessor

        Args:
            param_names (list[str]): List of parameter names (column names) to use as inputs.
            df (pd.DataFrame): The DataFrame containing the columns of parameter values and associated target value
            target (str): The name of the column containing target value within df
            folder_save_name (str): The name of the folder to be created that will contain med results (Default: "med_study")
        """
        # Validation: ensure all param_names are in df columns
        missing = [p for p in param_names if p not in df.columns]
        if missing:
            raise ValueError(f"The following param_names are not found in the DataFrame columns: {missing}")

        self.df = df
        self.target = target
        self.folder_save_name = folder_save_name
        self.param_names = param_names


    def prepare_data(self, split_frac: float, seed=42) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list]]:
        """
        Prepares input data for MED.

        Args:
            split_frac (float): Fraction of data to use for training.
            seed (int): Random seed for reproducibility.

        Returns:
            tuple: (train_df, test_df, parameters)
        """
        train_df, test_df = split_df(self.df, split_frac, seed)

        minimums = [self.df[param].min() for param in self.param_names]
        maximums = [self.df[param].max() for param in self.param_names]

        parameters = {
            "names": self.param_names,
            "minimums": minimums,
            "maximums": maximums
        }

        return train_df, test_df, parameters


    def run_med_discovery(self, train_df: pd.DataFrame, parameters: dict):
        """
        Runs MED symbolic regression to discover equations.

        Args:
            train_df (pd.DataFrame): Training dataset with labelled data points.
            parameters (dict): Dictionary containing parameter names and bounds.
        """
        target_values = train_df[self.target]

        # Create MED parameters
        med_params = medeq.create_parameters(
            parameters["names"],
            minimums=parameters["minimums"],
            maximums=parameters["maximums"]
        )

        # Create MED object
        med = medeq.MED(med_params, response_names=target_values.name, seed=200)

        # Add data
        med.augment(train_df[parameters["names"]], target_values)

        # Save results
        med.save(self.folder_save_name)

        # Discover equations
        med.discover(
            binary_operators=["+", "-", "*", "/", "^"],
            constraints={"^": (-1, 1)},  # (-1, 1) is the encoded constraint for "^" operator
            unary_operators=["exp", "log"]
        )


    def test_equations(self, test_df: pd.DataFrame, tmp_path=None) -> pd.DataFrame:
        """
        Tests MED results by evaluating equations on test data and computing relative errors.

        Args:
            test_df (pd.DataFrame): Testing dataset.
            tmp_path (str, optional): Path to MED result CSV file.

        Returns:
            pd.DataFrame: A DataFrame containing actual values and relative errors for each complexity.
        """
        eq_path = tmp_path if tmp_path else f"{self.folder_save_name}/hall_of_fame.csv"
        df_equations = pd.read_csv(eq_path)

        # Create symbolic functions for each equation using n parameters
        df_equations["Function"] = df_equations.apply(
            lambda row: create_function(row["Equation"], *self.param_names), axis=1
        )

        # Sort test data by all parameters (for consistency)
        test_df = test_df.sort_values(by=self.param_names)

        results = []
        for _, test_row in test_df.iterrows():
            result_row = {param: test_row[param] for param in self.param_names}
            result_row[self.target] = test_row[self.target]

            for _, eq_row in df_equations.iterrows():
                func = eq_row["Function"]
                complexity = eq_row["Complexity"]

                if func:
                    # Ensure correct parameter order
                    param_values = [test_row[param] for param in self.param_names]
                    pred = func(*param_values)
                    actual = test_row[self.target]

                    err = np.abs(actual - pred) / (np.abs(actual) + 1e-8)
                    result_row[f"Complexity {complexity} {self.target} p err"] = err

            results.append(result_row)

        return pd.DataFrame(results)


    def get_complexity_equation(self, complexity: int, tmp_path=None) -> str | None:
        """
        Fetches the equation from the hall_of_fame.csv file based on the given complexity.

        Args:
            complexity (int): The complexity level to find the equation for.

        Returns:
            str: The equation corresponding to the given complexity, or None if not found.
        """
        eq_path = tmp_path if tmp_path else f"{self.folder_save_name}/hall_of_fame.csv"
        df_equations = pd.read_csv(eq_path)

        equation_row = df_equations[df_equations["Complexity"] == complexity]

        return equation_row["Equation"].values[0] if not equation_row.empty else None