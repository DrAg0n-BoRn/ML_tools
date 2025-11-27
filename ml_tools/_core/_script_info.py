
def _script_info(all_data: list[str]):
    """
    List available names.
    """
    print("Available functions and objects:")
    for i, name in enumerate(all_data, start=1):
            print(f"{i} - {name}")
