import importlib.util
import sys

MODULES = {
    "yaml": "PyYAML",
    "requests": "requests",
    "jinja2": "Jinja2",
    "json_repair": "json-repair",
}


def main():
    missing = [
        package
        for module, package in MODULES.items()
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        return 0
    print(
        "Missing dependencies: "
        + ", ".join(missing)
        + ". Run: python -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
