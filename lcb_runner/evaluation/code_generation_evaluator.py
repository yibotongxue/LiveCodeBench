import asyncio

import numpy as np

from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness


async def evaluate_single_problem_async(
    problem_generations: list[str],
    sample: dict,
    debug: bool,
    timeout: int,
):
    """
    对单题的所有 generations 进行评测。
    每个 generation 串行评测。
    使用 multiprocessing.Process（与同步模式相同），用 asyncio.to_thread 异步等待。
    注意：并发控制由调用方通过 semaphore 管理。
    """
    res = []
    metadata = []
    for gen in problem_generations:
        # 用 asyncio.to_thread 把同步的 check_correctness 包装成 async
        # check_correctness 内部使用 multiprocessing.Process，与同步模式完全一致
        curr_res, curr_metadata = await asyncio.to_thread(
            check_correctness, sample, gen, timeout=timeout, debug=debug
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
