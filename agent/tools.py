"""
Pure-Python Tool Registry & Schema Generator (Built from Scratch)
Converts Python functions into typed JSON schemas for the Agent Brain.
"""

import inspect
import json
from typing import Any, Callable, Dict, List, Optional

class Tool:
    def __init__(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "").strip()
        self.schema = self._build_schema()

    def _build_schema(self) -> Dict[str, Any]:
        """Inspects function signature and generates standard JSON Schema."""
        sig = inspect.signature(self.func)
        properties = {}
        required = []

        type_map = {
            int: "integer",
            float: "number",
            str: "string",
            bool: "boolean",
            list: "array",
            dict: "object"
        }

        for param_name, param in sig.parameters.items():
            if param_name in ["self", "cls"]:
                continue

            param_type = type_map.get(param.annotation, "string")
            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter: {param_name}"
            }
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def execute(self, **kwargs) -> Any:
        sig = inspect.signature(self.func)
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_var_kwargs:
            return self.func(**kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return self.func(**filtered)

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, name: Optional[str] = None, description: Optional[str] = None):
        """Decorator to register any Python function as a tool."""
        def decorator(func: Callable):
            tool = Tool(func, name=name, description=description)
            self._tools[tool.name] = tool
            return func
        return decorator

    def add_tool(self, tool_func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        tool = Tool(tool_func, name=name, description=description)
        self._tools[tool.name] = tool

    def get_schemas(self) -> List[Dict[str, Any]]:
        return [tool.schema for tool in self._tools.values()]

    def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self._tools:
            return {"error": f"Tool '{name}' is not recognized."}
        try:
            result = self._tools[name].execute(**arguments)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())
