import ast
from pathlib import Path


ROOT = Path(__file__).parents[3]
RAYLIGHT = ROOT / "custom_nodes/raylight/src/raylight/diffusion_models/lightricks/xdit_context_parallel.py"


def _function(path, function_name):
    return next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == function_name)


def _calls(function, name):
    return [node for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name]


def test_ltx_stg_self_attention_skips_attention_call():
    function = _function(RAYLIGHT, "usp_cross_attn_forward")
    stg_branch = next(
        node for node in ast.walk(function)
        if isinstance(node, ast.If) and "stg_skip_self_attn" in ast.unparse(node.test)
    )

    assert "self_attn" in ast.unparse(stg_branch.test)
    assert any(isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "out" and ast.unparse(node.value) == "v" for node in stg_branch.body)
    assert not _calls(ast.Module(body=stg_branch.body, type_ignores=[]), "xfuser_optimized_attention")
    assert _calls(ast.Module(body=stg_branch.orelse, type_ignores=[]), "xfuser_optimized_attention")
