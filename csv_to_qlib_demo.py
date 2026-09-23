# -*- coding: utf-8 -*-
"""
CSV → Qlib 二进制 转换脚本（CLI，支持增量更新）
==================================================

⚠️  本脚本**不自己实现 bin 格式** — 全量/增量/修复 全部调用 qlib 官方类：
    scripts/dump_bin.py :: DumpDataAll / DumpDataUpdate / DumpDataFix

    官方类签名:
      DumpDataAll(     data_path, qlib_dir, freq='day', max_workers=16,
                       date_field_name='date', file_suffix='.csv',
                       symbol_field_name='symbol')
      DumpDataUpdate(  data_path, qlib_dir, ... 同上 ...)
      DumpDataFix(     data_path, qlib_dir, ... 同上 ...)

用法
----
python3 csv_to_qlib_demo.py --csv /data/csvs/ --qlib /data/qlib_bin
python3 csv_to_qlib_demo.py --csv /data/today.csv --qlib /data/qlib_bin
python3 csv_to_qlib_demo.py --demo          # 内置 mock + 验证

CSV 格式
--------
多文件模式  (--csv 传目录):
  data/
    ├── sh600000.csv   （文件名 = 股票代码，文件里**不要**带 symbol 列）
    ├── sh601318.csv
    每个文件列: date, open, high, low, close, volume

单文件模式  (--csv 传单个 .csv):
  all.csv  （一张大表，**必须**带 symbol 列）
  date, symbol, open, high, low, close, volume
  2024-01-02, sh600000, 10.0, ...

模式识别（自动）
---------------
  qlib_dir/ 不存在  →  DumpDataAll  (全量初始化)
  qlib_dir/ 已存在  →  DumpDataUpdate  (增量追加)
  --mode all|update|fix 可强制覆盖

官方 DumpDataAll 的一个限制
---------------------------
DumpDataAll 只支持"每文件 = 1 支股票"（symbol 从文件名取）。
如果用户用 --mode all 却给了单文件多股票 CSV，本脚本会**先按 symbol 拆
成多个临时文件**再喂给 DumpDataAll。DumpDataUpdate 本身两种格式都能吃。
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

# 让 qlib 官方脚本能被 import
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "scripts"))
from dump_bin import DumpDataAll, DumpDataFix, DumpDataUpdate  # noqa: E402


DEFAULT_FREQ = "day"


# ===========================================================================
# 【适配层】官方类的 CSV 格式桥接
# ===========================================================================
def _prepare_for_dump_all(csv_path: Path, tmp_root: Path) -> Path:
    """
    DumpDataAll 要求"每文件 = 1 支股票"。
    如果 csv_path 是目录 → 直接返回它（已经是多文件格式）。
    如果 csv_path 是单文件且有多 symbol → 按 symbol 拆成临时目录。
    """
    if csv_path.is_dir():
        return csv_path

    df = pd.read_csv(csv_path)
    if "symbol" not in df.columns:
        # 单文件没 symbol 列 —— 那就当作一支股票（文件名当 symbol）
        raise ValueError("单文件模式必须有 symbol 列，当前列: " + str(list(df.columns)))

    symbols = df["symbol"].nunique()
    if symbols == 1:
        # 只有 1 支股票，也得先拆（DumpDataAll 从文件名取 symbol）
        sym_dir = tmp_root / csv_path.stem
        sym_dir.mkdir(parents=True, exist_ok=True)
        sym = df["symbol"].iloc[0]
        df.drop(columns=["symbol"]).to_csv(sym_dir / f"{sym}.csv", index=False)
        return sym_dir

    # 多支 → 按 symbol 拆
    split_dir = tmp_root / "split_for_dump_all"
    split_dir.mkdir(parents=True, exist_ok=True)
    for sym, g in df.groupby("symbol"):
        g.drop(columns=["symbol"]).to_csv(split_dir / f"{sym}.csv", index=False)
    print(f"   单文件 {csv_path.name} 含 {symbols} 支股票 → 已拆到 {split_dir}")
    return split_dir


def run(mode: str, csv_path: str, qlib_dir: str, freq: str, max_workers: int):
    """入口：mode=auto|all|update|fix"""
    csv_p = Path(csv_path)
    qdir = Path(qlib_dir)

    if not csv_p.exists():
        raise FileNotFoundError(f"--csv 路径不存在: {csv_path}")

    if mode == "auto":
        mode = "update" if (qdir / "calendars" / f"{freq}.txt").exists() else "all"
        print(f"🔍 自动识别模式: {mode}  (qlib calendars 不存在={'是' if mode=='all' else '否'})")

    print(f"\n📂 CSV 输入: {csv_p}  ({'目录' if csv_p.is_dir() else '单文件'})")
    print(f"📦 Qlib 输出: {qdir}")
    print(f"⚙️  频率: {freq}")

    if mode == "all":
        # DumpDataAll 需要干净目录 + 多文件格式
        if qdir.exists():
            print(f"⚠️  全量模式：先清理已有 {qdir}")
            shutil.rmtree(qdir)
        with tempfile.TemporaryDirectory(prefix="dump_all_") as tmp:
            prepared = _prepare_for_dump_all(csv_p, Path(tmp))
            print(f"\n🆕 调用官方 DumpDataAll ...")
            DumpDataAll(
                data_path=str(prepared),
                qlib_dir=str(qdir),
                freq=freq,
                max_workers=max_workers,
            ).dump()
        print(f"✅ DumpDataAll 完成 → {qdir.resolve()}")

    elif mode == "update":
        if not (qdir / "calendars" / f"{freq}.txt").exists():
            raise FileNotFoundError(
                f"--mode update 但 qlib calendars 不存在: {qdir / 'calendars'}"
            )
        print(f"\n🔄 调用官方 DumpDataUpdate ...")
        DumpDataUpdate(
            data_path=str(csv_p),
            qlib_dir=str(qdir),
            freq=freq,
            max_workers=max_workers,
        ).dump()
        print(f"✅ DumpDataUpdate 完成")

    elif mode == "fix":
        if not (qdir / "calendars" / f"{freq}.txt").exists():
            raise FileNotFoundError(f"--mode fix 但 qlib 不存在: {qdir}")
        with tempfile.TemporaryDirectory(prefix="dump_fix_") as tmp:
            prepared = _prepare_for_dump_all(csv_p, Path(tmp))
            print(f"\n🛠️  调用官方 DumpDataFix ...")
            DumpDataFix(
                data_path=str(prepared),
                qlib_dir=str(qdir),
                freq=freq,
                max_workers=max_workers,
            ).dump()
        print(f"✅ DumpDataFix 完成")
    else:
        raise ValueError(f"未知 mode: {mode}")

    # 打印产物
    print("\n📁 产物:")
    for sub in ["calendars", "instruments", "features"]:
        p = qdir / sub
        if p.exists():
            n = len(list(p.rglob("*")))
            print(f"   {sub}/  → {n} 个文件")


# ===========================================================================
# 【内置 demo】
# ===========================================================================
def run_demo():
    ROOT = Path("/workspace/qlib_demo_data")
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir()

    import numpy as np

    np.random.seed(42)
    dates_a = list(pd.bdate_range("2024-01-02", periods=60))
    stock_bases = {"sh600000": 10.0, "sh601318": 45.0, "sz000001": 12.0}

    def _ohlcv(dates, base, seed):
        rng = np.random.RandomState(seed)
        rets = rng.randn(len(dates)) * 0.02
        close = base * np.cumprod(1 + rets)
        return pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "open":  (close + rng.randn(len(dates)) * 0.05).round(3),
            "high":  (close + abs(rng.randn(len(dates)) * 0.1)).round(3),
            "low":   (close - abs(rng.randn(len(dates)) * 0.1)).round(3),
            "close": close.round(3),
            "volume": np.random.randint(1_000_000, 10_000_000, len(dates)).astype(float),
        })

    # --- 批次 A: 60 天, 3 支股票 ---
    batch_a_frames = []
    for i, (sym, base) in enumerate(stock_bases.items()):
        df = _ohlcv(dates_a, base, i)
        df["symbol"] = sym
        batch_a_frames.append(df)
    batch_a = pd.concat(batch_a_frames, ignore_index=True)

    # ① 多文件格式（DumpDataAll 原生支持）
    multi_a = ROOT / "csv_multi"; multi_a.mkdir()
    for sym, g in batch_a.groupby("symbol"):
        g.drop(columns=["symbol"]).to_csv(multi_a / f"{sym}.csv", index=False)

    # ② 单文件格式（一张大表带 symbol）
    single_a = ROOT / "batchA_single.csv"
    batch_a.to_csv(single_a, index=False)

    # --- 批次 B: 75 天, 含 1 支新股票 sh600519 ---
    dates_b = list(pd.bdate_range("2024-04-01", periods=75))
    batch_b_frames = []
    for i, (sym, base) in enumerate({**stock_bases, "sh600519": 1700.0}.items()):
        df = _ohlcv(dates_b, base, i + 100)
        df["symbol"] = sym
        batch_b_frames.append(df)
    batch_b = pd.concat(batch_b_frames, ignore_index=True)
    single_b = ROOT / "batchB_single.csv"
    batch_b.to_csv(single_b, index=False)

    # ────────────────────────────────
    # Step 1: 全量初始化（多文件 → 官方 DumpDataAll 原生）
    # ────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1: 全量初始化  多文件  →  官方 DumpDataAll")
    print("=" * 60)
    qdir = ROOT / "qlib_bin"
    run(mode="auto", csv_path=str(multi_a), qlib_dir=str(qdir), freq=DEFAULT_FREQ, max_workers=2)

    # ────────────────────────────────
    # Step 2: 增量更新（单文件  →  官方 DumpDataUpdate，两种格式都支持）
    # ────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: 增量更新  单文件  →  官方 DumpDataUpdate")
    print("=" * 60)
    run(mode="auto", csv_path=str(single_b), qlib_dir=str(qdir), freq=DEFAULT_FREQ, max_workers=2)

    # ────────────────────────────────
    # Step 3: 单文件做全量（演示适配层拆文件）
    # ────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: 全量  但输入是单文件（演示适配层拆 symbol）")
    print("=" * 60)
    qdir2 = ROOT / "qlib_bin_from_single"
    run(mode="all", csv_path=str(single_a), qlib_dir=str(qdir2), freq=DEFAULT_FREQ, max_workers=2)

    # ────────────────────────────────
    # 验证：qlib D.features
    # ────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: qlib D.features() 验证")
    print("=" * 60)
    import qlib
    qlib.init(provider_uri={DEFAULT_FREQ: str(qdir)}, region="cn")
    from qlib.data import D

    res = D.features(
        ["sh600000", "sh601318", "sz000001", "sh600519"],
        ["$close", "$volume"],
        start_time="2024-01-02", end_time="2024-07-15",
        freq=DEFAULT_FREQ,
    )
    print(f"D.features shape={res.shape}")
    print(res.groupby(level=0).count().to_string())
    print(f"\n📊 sh600000 前 5 行:\n{res.loc['sh600000'].head()}")

    print("\n🎉 demo 完成。")


# ===========================================================================
# CLI 入口
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="CSV → qlib 二进制（全量/增量 直接调用官方 DumpDataAll/DumpDataUpdate）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
官方类说明（scripts/dump_bin.py）:
  DumpDataAll    — 全量初始化
  DumpDataUpdate — 增量追加（只加新日期）
  DumpDataFix    — 补缺失数据

模式识别（自动）:
  qlib_bin/calendars/day.txt 不存在 → DumpDataAll 全量
  qlib_bin/calendars/day.txt 存在   → DumpDataUpdate 增量
  --mode all|update|fix 可强制覆盖

CSV 格式:
  多文件模式 (--csv 目录): 每个文件 = 1 支股票, 文件名 = symbol, 无 symbol 列
  单文件模式 (--csv 文件): 一张大表, 必须带 symbol 列

示例:
  python3 csv_to_qlib_demo.py --demo
  python3 csv_to_qlib_demo.py --csv /data/csvs/  --qlib /data/qlib_bin
  python3 csv_to_qlib_demo.py --csv /data/today.csv --qlib /data/qlib_bin
""",
    )
    ap.add_argument("--csv",    help="输入 csv 文件或目录")
    ap.add_argument("--qlib",   help="qlib 二进制输出目录")
    ap.add_argument("--freq",   default=DEFAULT_FREQ)
    ap.add_argument("--mode",   default="auto", choices=["auto", "all", "update", "fix"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--demo",   action="store_true")
    args = ap.parse_args()

    if args.demo:
        run_demo()
        return
    if not args.csv or not args.qlib:
        ap.error("必须同时指定 --csv 和 --qlib，或用 --demo")

    run(args.mode, args.csv, args.qlib, args.freq, args.workers)


if __name__ == "__main__":
    main()
