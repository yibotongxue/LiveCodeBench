import asyncio
import json
import os
import sys
import tempfile

import numpy as np


async def run_test_in_subprocess(sample, generation, timeout, debug=False):
    """
    用 asyncio.create_subprocess_exec 启动独立 Python 子进程执行 run_test。
    完全不用 multiprocessing。
    """
    # 准备测试数据
    test_data = {
        "sample": sample,
        "generation": generation,
        "timeout": timeout,
        "debug": debug,
    }

    # 创建临时数据文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(test_data, f)
        data_path = f.name

    # 获取子进程脚本路径（同一目录下的 eval_subprocess_runner.py）
    script_dir = os.path.dirname(__file__)
    script_path = os.path.join(script_dir, "eval_subprocess_runner.py")

    try:
        # 启动子进程
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            script_path,
            data_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # 计算全局超时（基于测试用例数量）
        n_inputs = len(json.loads(sample["input_output"]).get("inputs", []))
        global_timeout = (timeout + 1) * n_inputs + 5

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=global_timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            in_outs = json.loads(sample["input_output"])
            return [-1] * len(in_outs["inputs"]), {"error": "global timeout"}

        if proc.returncode != 0:
            in_outs = json.loads(sample["input_output"])
            return [-1] * len(in_outs["inputs"]), {
                "error": stderr.decode()[:500],
                "error_code": -5,
                "error_message": "SubprocessError",
            }

        result = json.loads(stdout.decode())
        return result["res"], result["metadata"]

    finally:
        os.unlink(data_path)


async def evaluate_single_problem_async(
    problem_generations: list[str],
    sample: dict,
    debug: bool,
    timeout: int,
    eval_semaphore: asyncio.Semaphore,
):
    """
    对单题的所有 generations 进行评测。
    每个 generation 串行评测，但用 eval_semaphore 限制同时运行的子进程数。
    """
    res = []
    metadata = []
    for gen in problem_generations:
        async with eval_semaphore:
            curr_res, curr_metadata = await run_test_in_subprocess(
                sample, gen, timeout=timeout, debug=debug
            )
            # 类型处理（numpy 数组转 Python 类型）
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
        res.append(curr_res)
        metadata.append(curr_metadata)
    return res, metadata
