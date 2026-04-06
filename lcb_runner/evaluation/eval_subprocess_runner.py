#!/usr/bin/env python3
"""
独立子进程脚本，用于在 asyncio.create_subprocess_exec 中执行测试。
接收数据文件路径作为命令行参数，输出 JSON 结果到 stdout。
"""
import json
import sys
import os

# 确保能找到 lcb_runner 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcb_runner.evaluation.testing_util import run_test


def main():
    data_path = sys.argv[1]
    with open(data_path, "r") as f:
        data = json.load(f)

    sample = data["sample"]
    generation = data["generation"]
    timeout = data["timeout"]
    debug = data.get("debug", False)

    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)

    # numpy 类型需要序列化
    def serialize(obj):
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    print(json.dumps({"res": res, "metadata": metadata}, default=serialize))


if __name__ == "__main__":
    main()
