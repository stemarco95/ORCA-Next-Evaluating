import os
import inspect
import importlib

from core.base_module import BaseModule
from core.audit_logger import AuditLogger


def snake_to_pascal(name: str) -> str:
    """Convert snake_case to PascalCase"""
    return "".join(part.capitalize() for part in name.split("_"))


def auto_load_modules(directory: str = "modules") -> dict:
    """
    Dynamically import all module classes from the given directory and subdirectories.
    Assumes:
        file name: snake_case
        class name: PascalCase version of it.

    Returns:
        dict: Mapping type_name -> class object
    """
    registry = {}

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".py") and not filename.startswith("_"):
                mod_name = filename[:-3]
                class_name = snake_to_pascal(mod_name)
                
                # Build module path from directory structure
                rel_path = os.path.relpath(root, ".")
                module_path = rel_path.replace(os.sep, ".") + "." + mod_name

                try:
                    module = importlib.import_module(module_path)
                    cls = getattr(module, class_name, None)

                    if inspect.isclass(cls) and issubclass(cls, BaseModule):
                        registry[class_name] = cls
                        AuditLogger.log_message(f"Module loaded: {class_name} from {module_path}")
                    else:
                        AuditLogger.log_event(
                            "module_warning",
                            msg=f"Expected class '{class_name}' not found in '{module_path}'"
                        )

                except Exception as e:
                    AuditLogger.log_event(
                        "module_error",
                        module=module_path,
                        error=str(e)
                    )

    return registry