from lcb_runner.lm_styles import LMStyle


_PYTHON_CODE_STARTERS = (
    "def ", "class ", "import ", "from ", "# ", "print(", "if ", "for ", "while ",
    "with ", "try:", "@", "="
)


def _find_code_block(lines: list[str]) -> str | None:
    """Try to extract code from markdown ```python ... ``` or ``` ... ``` blocks."""
    # First try ```python
    py_indices = [i for i, line in enumerate(lines) if line.strip().startswith("```python")]
    if len(py_indices) >= 1:
        start = py_indices[0]
        # Find the matching closing ```
        for i in range(start + 1, len(lines)):
            if lines[i].strip() == "```":
                return "\n".join(lines[start + 1 : i])
    # Then try plain ```
    indices = [i for i, line in enumerate(lines) if line.strip().startswith("```")]
    if len(indices) >= 2:
        return "\n".join(lines[indices[-2] + 1 : indices[-1]])
    return None


def _find_python_code_fallback(lines: list[str]) -> str | None:
    """Fallback: look for Python code starters (def, class, import, etc.)."""
    start_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(_PYTHON_CODE_STARTERS):
            start_idx = i
            break
    if start_idx is None:
        return None

    # Try to trim trailing explanation text after the code block.
    # Heuristic: from the end, skip empty lines. If we hit a line that looks like
    # plain English explanation (long, no code indicators), truncate there.
    end_idx = len(lines)
    seen_non_empty = False
    for i in range(len(lines) - 1, start_idx - 1, -1):
        stripped = lines[i].strip()
        if not stripped:
            if seen_non_empty:
                end_idx = i + 1
                break
            continue
        seen_non_empty = True
        # If this looks like a sentence (ends with . or ! or ?) and is fairly long,
        # and doesn't look like code, it's probably trailing explanation.
        if (stripped.endswith((".", "!", "?")) and len(stripped) > 40
                and not stripped.startswith(("#", "print(", "return ", "    ", "\t", "if ", "for ", "while ", "def ", "class ", "import ", "from "))):
            end_idx = i
            break
        end_idx = i + 1
        break

    return "\n".join(lines[start_idx:end_idx])


def extract_code(model_output: str, lmstyle: LMStyle):
    outputlines = model_output.split("\n")
    if lmstyle == LMStyle.CodeLLaMaInstruct:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
        if len(indexlines) >= 2:
            return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])
        # Fall through to general extraction
    elif lmstyle == LMStyle.GenericBase:
        return model_output.strip()

    # General extraction pipeline
    # 1. Try markdown code blocks
    code = _find_code_block(outputlines)
    if code is not None:
        return code.strip()

    # 2. Fallback: look for Python code patterns
    code = _find_python_code_fallback(outputlines)
    if code is not None:
        return code.strip()

    # 3. Last resort: return whole output
    return model_output.strip()


def extract_test_output_code(model_output: str, lmstyle: LMStyle = None):
    outputlines = model_output.split("\n")
    # find the last line startwith assert...
    indexlines = [i for i, line in enumerate(outputlines) if line.startswith("assert")]
    if indexlines:
        return outputlines[indexlines[-1]]
    if lmstyle and lmstyle == LMStyle.CodeLLaMaInstruct:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
    else:
        # first try to extract ```python if not then try ```
        indexlines = [
            i
            for i, line in enumerate(outputlines)
            if "```python" in line or "```Python" in line
        ]
        if indexlines:
            start_index = indexlines[0]
        else:
            start_index = None
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
        if start_index is not None:
            indexlines = [i for i in indexlines if i > start_index]
            indexlines = [start_index] + indexlines

    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])


def extract_execution_code(model_output: str, lmstyle: LMStyle, cot: bool = False):
    if cot:
        if "[ANSWER]" in model_output:
            model_output = model_output.split("[ANSWER]")[1].strip()
    if "==" in model_output:
        model_output = model_output.split("==")[1].strip()
    if "[/ANSWER]" in model_output:
        model_output = model_output.split("[/ANSWER]")[0].strip()
    else:
        model_output = model_output.split("\n")[0].strip()
    return model_output.strip()
