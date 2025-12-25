
def _imprimir_disponibles(all_data: list[str]):
    """
    List available names in namespace.
    """
    print("Available functions and objects:")
    for i, name in enumerate(all_data, start=1):
            print(f"{i} - {name}")
