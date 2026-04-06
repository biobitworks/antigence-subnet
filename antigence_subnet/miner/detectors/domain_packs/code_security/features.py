"""AST-based code security feature extraction.

Uses Python's ast module to detect insecure code patterns: dangerous function
calls, string concatenation in function arguments (SQL injection), hardcoded
credentials, risky imports, deep nesting, eval/exec usage.
"""

import ast

# Functions considered dangerous for arbitrary code/command execution
_DANGEROUS_FUNCTIONS = frozenset({
    "system", "popen",          # os.system, os.popen
    "call", "run", "Popen",    # subprocess.*
    "eval", "exec",            # built-in eval/exec
    "__import__", "compile",   # dynamic import/compile
})

# Qualified dangerous calls (module.function)
_DANGEROUS_QUALIFIED = frozenset({
    "os.system", "os.popen",
    "subprocess.call", "subprocess.run", "subprocess.Popen",
})

# Risky modules to import
_RISKY_MODULES = frozenset({
    "pickle", "shelve", "marshal", "ctypes", "subprocess", "os",
})

# Credential-related variable name fragments
_CREDENTIAL_NAMES = frozenset({
    "password", "secret", "key", "token", "api_key",
})


def _get_call_name(node: ast.Call) -> str:
    """Extract the function name from a Call node.

    Returns the qualified name (e.g., 'os.system') or simple name (e.g., 'eval').
    Returns '' if the name cannot be determined.
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        # Handle one level of attribute access (e.g., os.system)
        if isinstance(func.value, ast.Name):
            return f"{func.value.id}.{func.attr}"
        return func.attr
    return ""


def _max_depth(node: ast.AST, current: int = 0) -> int:
    """Compute the maximum depth of an AST tree."""
    max_d = current
    for child in ast.iter_child_nodes(node):
        d = _max_depth(child, current + 1)
        if d > max_d:
            max_d = d
    return max_d


def _has_string_concat_arg(node: ast.Call) -> bool:
    """Check if any argument to a Call is a string concatenation (BinOp with Add)."""
    for arg in list(node.args) + [kw.value for kw in node.keywords]:
        if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
            return True
    return False


def extract_code_security_features(code: str) -> dict[str, float]:
    """Extract 7 security-relevant features from Python code using AST analysis.

    Args:
        code: Python source code string to analyze.

    Returns:
        Dict with 7 float features:
            - dangerous_call_count: calls to dangerous functions
            - string_concat_in_call: function calls with string concatenation args
            - hardcoded_credential_score: assignments of string literals to credential vars
            - import_risk_score: imports of risky modules
            - ast_node_depth: maximum AST tree depth
            - exec_eval_usage: specific count of eval/exec calls
            - total_function_calls: total ast.Call node count
    """
    defaults = {
        "dangerous_call_count": 0.0,
        "string_concat_in_call": 0.0,
        "hardcoded_credential_score": 0.0,
        "import_risk_score": 0.0,
        "ast_node_depth": 0.0,
        "exec_eval_usage": 0.0,
        "total_function_calls": 0.0,
    }

    if not code or not code.strip():
        return defaults

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return defaults

    dangerous_call_count = 0
    string_concat_in_call = 0
    hardcoded_credential_score = 0
    import_risk_score = 0
    exec_eval_usage = 0
    total_function_calls = 0

    for node in ast.walk(tree):
        # Count function calls and dangerous patterns
        if isinstance(node, ast.Call):
            total_function_calls += 1
            name = _get_call_name(node)

            # Check dangerous calls
            simple_name = name.split(".")[-1] if "." in name else name
            if name in _DANGEROUS_QUALIFIED or simple_name in _DANGEROUS_FUNCTIONS:
                dangerous_call_count += 1

            # Check eval/exec specifically
            if simple_name in ("eval", "exec"):
                exec_eval_usage += 1

            # Check string concatenation in call args
            if _has_string_concat_arg(node):
                string_concat_in_call += 1

        # Check hardcoded credentials
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                target_name = ""
                if isinstance(target, ast.Name):
                    target_name = target.id.lower()
                elif isinstance(target, ast.Attribute):
                    target_name = target.attr.lower()

                if (
                    target_name
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    for cred_name in _CREDENTIAL_NAMES:
                        if cred_name in target_name:
                            hardcoded_credential_score += 1
                            break

        # Check risky imports
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".")[0]
                if module_name in _RISKY_MODULES:
                    import_risk_score += 1

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split(".")[0]
                if module_name in _RISKY_MODULES:
                    import_risk_score += 1

    ast_node_depth = float(_max_depth(tree))

    return {
        "dangerous_call_count": float(dangerous_call_count),
        "string_concat_in_call": float(string_concat_in_call),
        "hardcoded_credential_score": float(hardcoded_credential_score),
        "import_risk_score": float(import_risk_score),
        "ast_node_depth": ast_node_depth,
        "exec_eval_usage": float(exec_eval_usage),
        "total_function_calls": float(total_function_calls),
    }


def code_from_sample(sample: dict) -> str:
    """Extract code from a sample dict.

    Prefers sample['code'] if present, falls back to sample['output'], then ''.

    Args:
        sample: Sample dict potentially containing 'code' and/or 'output' keys.

    Returns:
        Code string to analyze.
    """
    if sample.get("code"):
        return sample["code"]
    if sample.get("output"):
        return sample["output"]
    return ""
