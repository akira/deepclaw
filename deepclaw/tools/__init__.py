"""Tool plugin discovery for DeepClaw.

Each module in this package is a tool plugin. A plugin module must export:
  - available() -> bool    — True if deps and env vars are present
  - get_tools() -> list    — returns tool callables/dicts for create_deep_agent

The discover_tools() function scans all plugin modules and collects tools
from those that are available.
"""

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)


def discover_tools() -> list:
    """Scan plugin modules and return tools from all available plugins."""
    tools = []
    package = importlib.import_module(__name__)

    for finder, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
        if module_name.startswith("_"):
            continue
        fqn = f"{__name__}.{module_name}"
        try:
            mod = importlib.import_module(fqn)
        except Exception:
            logger.warning("Failed to import tool plugin %s", fqn, exc_info=True)
            continue

        available_fn = getattr(mod, "available", None)
        get_tools_fn = getattr(mod, "get_tools", None)

        if available_fn is None or get_tools_fn is None:
            logger.debug("Skipping %s: missing available() or get_tools()", module_name)
            continue

        try:
            if available_fn():
                plugin_tools = get_tools_fn()
                tools.extend(plugin_tools)
                logger.info("Loaded %d tools from plugin %s", len(plugin_tools), module_name)
            else:
                logger.debug("Plugin %s not available, skipping", module_name)
        except Exception:
            logger.warning("Error loading plugin %s", module_name, exc_info=True)

    return tools
