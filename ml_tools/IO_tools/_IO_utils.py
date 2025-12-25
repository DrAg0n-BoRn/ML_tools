import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Union

from ..path_manager import make_fullpath
from .._core import get_logger


_LOGGER = get_logger("IO tools")


__all__ = [
    "compare_lists",
    "_RobustEncoder"
]


class _RobustEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle non-serializable objects.

    This handles:
    1.  `type` objects (e.g., <class 'int'>) which result from
        `check_type_only=True`.
    2.  Any other custom class or object by falling back to its
        string representation.
    """
    def default(self, o):
        if isinstance(o, type):
            return str(o)
        try:
            return super().default(o)
        except TypeError:
            return str(o)


def compare_lists(
    list_A: list,
    list_B: list,
    save_dir: Union[str, Path],
    strict: bool = False,
    check_type_only: bool = False
) -> dict:
    """
    Compares two lists and saves a JSON report of the differences.

    Args:
        list_A (list): The first list to compare.
        list_B (list): The second list to compare.
        save_dir (str | Path): The directory where the resulting report will be saved.
        strict (bool):
            - If False: Performs a "bag" comparison. Order does not matter, but duplicates do.
            - If True: Performs a strict, positional comparison.
            
        check_type_only (bool):
            - If False: Compares items using `==` (`__eq__` operator).
            - If True: Compares only the `type()` of the items.

    Returns:
        dict: A dictionary detailing the differences. (saved to `save_dir`).
    """
    MISSING_A_KEY = "missing_in_A"
    MISSING_B_KEY = "missing_in_B"
    MISMATCH_KEY = "mismatch"
    
    results: dict[str, list] = {MISSING_A_KEY: [], MISSING_B_KEY: []}
    
    # make directory
    save_path = make_fullpath(input_path=save_dir, make=True, enforce="directory")

    if strict:
        # --- STRICT (Positional) Mode ---
        results[MISMATCH_KEY] = []
        sentinel = object()

        if check_type_only:
            compare_func = lambda a, b: type(a) == type(b)
        else:
            compare_func = lambda a, b: a == b

        for index, (item_a, item_b) in enumerate(
            zip_longest(list_A, list_B, fillvalue=sentinel)
        ):
            if item_a is sentinel:
                results[MISSING_A_KEY].append({"index": index, "item": item_b})
            elif item_b is sentinel:
                results[MISSING_B_KEY].append({"index": index, "item": item_a})
            elif not compare_func(item_a, item_b):
                results[MISMATCH_KEY].append(
                    {
                        "index": index,
                        "list_A_item": item_a,
                        "list_B_item": item_b,
                    }
                )

    else:
        # --- NON-STRICT (Bag) Mode ---
        if check_type_only:
            # Types are hashable, we can use Counter (O(N))
            types_A_counts = Counter(type(item) for item in list_A)
            types_B_counts = Counter(type(item) for item in list_B)

            diff_A_B = types_A_counts - types_B_counts
            for item_type, count in diff_A_B.items():
                results[MISSING_B_KEY].extend([item_type] * count)

            diff_B_A = types_B_counts - types_A_counts
            for item_type, count in diff_B_A.items():
                results[MISSING_A_KEY].extend([item_type] * count)

        else:
            # Items may be unhashable. Use O(N*M) .remove() method
            temp_B = list(list_B)
            missing_in_B = []

            for item_a in list_A:
                try:
                    temp_B.remove(item_a)
                except ValueError:
                    missing_in_B.append(item_a)
            
            results[MISSING_A_KEY] = temp_B
            results[MISSING_B_KEY] = missing_in_B

    # --- Save the Report ---
    try:
        full_path = save_path / "list_comparison.json"

        # Write the report dictionary to the JSON file
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, cls=_RobustEncoder)
            
    except Exception as e:
        _LOGGER.error(f"Failed to save comparison report to {save_path}: \n{e}")

    return results

