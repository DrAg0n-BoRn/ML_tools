import polars as pl
from pathlib import Path
from typing import Union, Optional, Any, Callable

from ..utilities import load_dataframe, save_dataframe_filename

from ..keys._keys import MagicWords
from ..path_manager import make_fullpath
from .._core import get_logger


_LOGGER = get_logger("Dragon ETL")


__all__ = [
    "DragonTransformRecipe",
    "DragonProcessor",
]


class DragonTransformRecipe:
    """
    A builder class for creating a data transformation recipe.

    This class provides a structured way to define a series of transformation
    steps, with validation performed at the time of addition. It is designed
    to be passed to a `DragonProcessor`.
    
    Use the method `add()` to add recipes.
    """
    def __init__(self):
        self._steps: list[dict[str, Any]] = []
        self._added_input_columns: set[str] = set()

    def add(
        self,
        input_col_name: str,
        transform: Optional[Callable] = None,
        output_col_names: Optional[Union[str, list[str]]] = None
    ) -> "DragonTransformRecipe":
        """
        Adds a new transformation step to the recipe.

        Args:
            input_col_name: The name of the column from the source DataFrame.
            transform: The transformation to apply. 
                - If None and `output_col_names` is None, the column is kept as is.
                - If None and `output_col_names` is a string, performs a simple rename.
                - If callable, must accept a `pl.Series` as the only parameter and return either a `pl.Series` or `pl.DataFrame`.
            output_col_names: The desired name(s) for the output column(s).
                - A string for a 1-to-1 mapping.
                - A list of strings for a 1-to-many mapping.
                - A string prefix for 1-to-many mapping.
                - If None, the input name is used for 1-to-1 transforms, or the transformer's default names are used for 1-to-many.

        Returns:
            The instance of the recipe itself to allow for method chaining.
        """
        # --- Validation ---
        if not isinstance(input_col_name, str) or not input_col_name:
            _LOGGER.error("'input_col' must be a non-empty string.")
            raise TypeError()
        
        # block duplicate input columns
        if input_col_name in self._added_input_columns:
            _LOGGER.error(f"A transformation for input column '{input_col_name}' already exists in the recipe. Only one transformation per column is allowed.\nFor multiple transformations, consider creating a custom function that chains them together or using several DragonProcessor pipeline stages.")
            raise ValueError()
        
        # Resolve the transformation and output column names based on the provided arguments
        resolved_transform = None
        if transform is None and output_col_names is None:
            resolved_transform = MagicWords.RENAME
            output_col_names = input_col_name
        elif transform is None:
            if not isinstance(output_col_names, str):
                _LOGGER.error("For a rename operation, 'output_col_names' must be a string.")
                raise TypeError()
            resolved_transform = MagicWords.RENAME
        elif not isinstance(transform, Callable):
            _LOGGER.error("'transform' must be a callable function or None.")
            raise TypeError()
        else:
            resolved_transform = transform
        
        # --- Add Step ---
        step = {
            "input_col": input_col_name,
            "output_col": output_col_names,
            "transform": resolved_transform,
        }
        self._steps.append(step)
        self._added_input_columns.add(input_col_name)
        
        return self  # Allow chaining: recipe.add(...).add(...)

    def __iter__(self):
        """Allows the class to be iterated over, like a list."""
        return iter(self._steps)

    def __len__(self):
        """Allows the len() function to be used on an instance."""
        return len(self._steps)
    
    def recipe_status(self) -> dict[str, dict[str, Union[str, list[str]]]]:
        """
        Returns a dictionary summarizing the transformation steps in the recipe.
        
        The keys are the input column names, and the values are dictionaries containing:
            - "output_columns": The specified output column names or a description of how they are determined
            - "transform": The name of the transformation function or a description of the action taken
        """
        mapping = {}
        for step in self._steps:
            input_col: str = step["input_col"]
            output_col: Union[str, list[str], None] = step["output_col"]
            transform: Union[Callable, str] = step["transform"]
            
            if transform == MagicWords.RENAME:
                transform_name = "No transformation with optional renaming"
                out_cols = output_col if output_col is not None else input_col
            else:
                # __name__ for functions, fallback to class name for callables
                transform_name: str = getattr(transform, "__name__", type(transform).__name__)
                out_cols = output_col if output_col is not None else "Output column names determined by the Transformation"
            
            mapping[input_col] = {
                "output_columns": out_cols,
                "transform": transform_name
            }

        return mapping


class DragonProcessor:
    """
    Transforms a Polars DataFrame based on a provided `DragonTransformRecipe` object.
    
    Use the methods `transform()` or `load_transform_save()`.
    """
    def __init__(self, recipe: DragonTransformRecipe):
        """
        Initializes the DragonProcessor with a transformation recipe.

        Args:
            recipe: An instance of the `DragonTransformRecipe` class that has
                    been populated with transformation steps.
        """
        if not isinstance(recipe, DragonTransformRecipe):
            _LOGGER.error("The recipe must be an instance of DragonTransformRecipe.")
            raise TypeError()
        if len(recipe) == 0:
            _LOGGER.error("The recipe cannot be empty.")
            raise ValueError()
        self._recipe = recipe

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Applies the transformation recipe to the input DataFrame.
        """
        processed_columns = []
        # Recipe object is iterable
        for step in self._recipe:
            input_col_name = step["input_col"]
            output_col_spec = step["output_col"]
            transform_action = step["transform"]

            if input_col_name not in df.columns:
                _LOGGER.error(f"Input column '{input_col_name}' not found in DataFrame.")
                raise ValueError()

            input_series = df.get_column(input_col_name)

            if transform_action == MagicWords.RENAME:
                processed_columns.append(input_series.alias(output_col_spec))
                continue

            if isinstance(transform_action, Callable):
                result = transform_action(input_series)

                if isinstance(result, pl.Series):
                    # Default to input name if spec is None
                    output_name = output_col_spec if output_col_spec is not None else input_col_name
                    
                    if not isinstance(output_name, str):
                        _LOGGER.error(f"Function for '{input_col_name}' returned a Series but 'output_col' must be a string or None.")
                        raise TypeError()
                    processed_columns.append(result.alias(output_name))
                
                elif isinstance(result, pl.DataFrame):
                    # 1. Handle None in output names
                    if output_col_spec is None:
                        # Use the column names generated by the transformer directly
                        processed_columns.extend(result.get_columns())
                    
                    # 2. Handle list-based renaming
                    elif isinstance(output_col_spec, list):
                        if len(result.columns) != len(output_col_spec):
                            _LOGGER.error(f"Mismatch in '{input_col_name}': function produced {len(result.columns)} columns, but recipe specifies {len(output_col_spec)} output names.")
                            raise ValueError()
                        
                        renamed_df = result.rename(dict(zip(result.columns, output_col_spec)))
                        processed_columns.extend(renamed_df.get_columns())
                    
                    # 3. Global logic for adding a single prefix to all columns.
                    elif isinstance(output_col_spec, str):
                        prefix = output_col_spec
                        new_names = {}
                        
                        for col in result.columns:
                            # Case 1: Transformer's output column name contains the input name.
                            # Action: Replace the input name with the desired prefix.
                            # Example: input='color', output='color_red', prefix='spec' -> 'spec_red'
                            # if input_col_name in col:
                            if col.startswith(input_col_name):
                                new_names[col] = col.replace(input_col_name, prefix, 1)
                            
                            # Case 2: Transformer's output is an independent name.
                            # Action: Prepend the prefix to the output name.
                            # Example: input='ratio', output='A_B', prefix='spec' -> 'spec_A_B'
                            else:
                                new_names[col] = f"{prefix}_{col}"
                                
                        renamed_df = result.rename(new_names)
                        processed_columns.extend(renamed_df.get_columns())    
                    
                    
                    else:
                        _LOGGER.error(f"Function for '{input_col_name}' returned a DataFrame, so 'output_col' must be a list of names, a string prefix, or None.")
                        raise TypeError()
                
                else:
                    _LOGGER.error(f"Function for '{input_col_name}' returned an unexpected type: {type(result)}.")
                    raise TypeError()
            
            else: # This case is unlikely due to builder validation.
                _LOGGER.error(f"Invalid 'transform' action for '{input_col_name}': {transform_action}")
                raise TypeError()

        if not processed_columns:
            _LOGGER.error("The transformation resulted in an empty DataFrame.")
            return pl.DataFrame()
        
        _LOGGER.info(f"Processed dataframe with {len(processed_columns)} columns.")

        return pl.DataFrame(processed_columns)
    
    def load_transform_save(self, input_path: Union[str,Path], output_path: Union[str,Path]):
        """
        Convenience wrapper for the transform method that includes automatic dataframe loading and saving.
        """
        # Validate paths
        in_path = make_fullpath(input_path, enforce="file")
        out_path = make_fullpath(output_path, make=True, enforce="file")
        
        # load df
        df, _ = load_dataframe(df_path=in_path, kind="polars", all_strings=True)
        
        # Process
        df_processed = self.transform(df)
        
        # save processed df
        save_dataframe_filename(df=df_processed, save_dir=out_path.parent, filename=out_path.name)
        
    def __str__(self) -> str:
        """
        Provides a detailed, human-readable string representation of the
        entire processing pipeline.
        """
        header = "DragonProcessor Pipeline"
        divider = "-" * len(header)
        num_steps = len(self._recipe)
        
        lines = [
            header,
            divider,
            f"Number of steps: {num_steps}\n"
        ]

        if num_steps == 0:
            lines.append("No transformation steps defined.")
            return "\n".join(lines)

        for i, step in enumerate(self._recipe, 1):
            transform_action = step["transform"]
            
            # Get a clean name for the transformation action
            if transform_action == MagicWords.RENAME: # "rename"
                transform_name = "Rename"
            else:
                # This works for both functions and class instances
                transform_name = type(transform_action).__name__

            lines.append(f"[{i}] Input: '{step['input_col']}'")
            lines.append(f"    - Transform: {transform_name}")
            lines.append(f"    - Output(s): {step['output_col']}")
            if i < num_steps:
                lines.append("") # Add a blank line between steps

        return "\n".join(lines)

    def inspect(self) -> None:
        """
        Prints the detailed string representation of the pipeline to the console.
        """
        print(self)

