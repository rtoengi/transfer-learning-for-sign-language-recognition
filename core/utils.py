from pathlib import Path


def package_path(package):
    """Returns the location of the passed package.

    Arguments:
        package: A package object.

    Returns:
        An absolute Path object pointing to the package's location.
    """
    return Path(package.__path__[0])
