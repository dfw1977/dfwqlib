# -*- coding: utf-8 -*-
"""
CSV → Qlib 二进制 转换脚本（CLI）
=================================

用法
----
# 1) 多文件模式：data_path 是一个目录，目录下每支股票一个 csv
python3 csv_to_qlib_demo.py --csv /data/csv_dir        --qlib /data/qlib_bin

# 2) 单文件模式：data_path 是一个 csv 文件，里面包含所有股票（必须有 symbol 列）
python3 csv_to_qlib_demo.py --csv /data/all.csv       --qlib /data/qlib_bin

# 3) 先跑 demo 数据（自动生成 mock csv + 转 qlib + 用 D.features 验证）
python3 csv_to_qlib_demo.py --demo

两个模式真正的唯一区别
----------------------
data_path 传「目录」还是「文件」：

  - 目录 (多文件)：脚本 glob(*.csv)，每个文件 = 一支股票
                  文件名 = 股票代码 (如 sh600000.csv)
                  也可以 CSV 里自带 symbol 列

  - 文件 (单文件)：脚本直接 pd.read_csv(one_file)
                  CSV 里必须有 symbol 列来区分股票

一旦 DataFrame 读进来之后，后续处理完全一样：
  去重 → 拼全局日历 → 每支股票 reindex 对齐 → 写 bin → 写 instruments

qlib bin 格式
-------------
features/<symbol>/<field>.day.bin =
    ┌────────────────────┬──────────────────────────────────┐
    │ float32 date_index  │ float32 × N  数据点 (缺失填 NaN)   │
    └────────────────────┴──────────────────────────────────┘
date_index = 这支股票第一个交易日在「全局日历」里的索引 (0-based)

目录结构
--------
qlib_bin/
├── calendars/day.txt       ← 全局交易日，每行一个 YYYY-MM-DD
├── instruments/all.txt     ← 每支股票存续期: symbol<TAB>start<TAB>end
└── features/
    ├── sh600000/
    │   ├── open.day.bin
    │   └── close.day.bin
    └── sz000001/
        └── ...
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 默认常量（改这里，也可以用 CLI 参数覆盖）
# ---------------------------------------------------------------------------
DEFAULT_FREQ = "day"
DEFAULT_MARKET = "all"
DEFAULT_FIELDS = ["open", "high", "low", "close", "volume"]


# ===========================================================================
# 【1 读数据】根据 data_path 类型，返回一张大表 (symbol, date, OHLCV...)
# ===========================================================================
def load_csv(data_path: str) -> pd.DataFrame:
    """
    data_path 可以是：
      - 一个 csv 文件   → 单文件模式，CSV 必须有 symbol 列
      - 一个目录       → 多文件模式，目录下 *.csv 每支股票一个文件
    """
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"--csv 路径不存在: {data_path}")

    if p.is_file():
        # ============== 单文件模式 ==============
        print(f"📄 单文件模式: {p}")
        df = pd.read_csv(p)
        if "symbol" not in df.columns:
            raise ValueError(
                f"单文件模式必须有 symbol 列，当前列: {list(df.columns)}"
            )
        print(f"   shape={df.shape}  symbols={df['symbol'].nunique()}")
        return df

    elif p.is_dir():
        # ============== 多文件模式 ==============
        csvs = sorted(p.glob("*.csv")) + sorted(p.glob("*.CSV"))
        if not csvs:
            raise FileNotFoundError(f"目录下没有 csv: {p}")
        print(f"📁 多文件模式: {p}  ({len(csvs)} 个 csv)")

        frames = []
        for fp in csvs:
            df = pd.read_csv(fp)
            # 文件名 stem 就是股票代码（如 sh600000.csv → sh600000）
            symbol_from_fname = fp.stem
            if "symbol" not in df.columns:
                df["symbol"] = symbol_from_fname
            else:
                # 有 symbol 列，但如果里面全空就用文件名补上
                df["symbol"] = df["symbol"].fillna(symbol_from_fname)
            frames.append(df)
        big = pd.concat(frames, ignore_index=True)
        print(f"   合并后 shape={big.shape}  symbols={big['symbol'].nunique()}")
        return big

    else:
        raise ValueError(f"--csv 既不是文件也不是目录: {data_path}")


# ===========================================================================
# 【2 转 qlib】一张大表 → calendars + instruments + features/*.bin
# ===========================================================================
def to_qlib_bin(all_df: pd.DataFrame, qlib_dir: str,
                freq: str = DEFAULT_FREQ,
                market: str = DEFAULT_MARKET,
                fields=None):
    fields = fields or DEFAULT_FIELDS
    qdir = Path(qlib_dir)
    if qdir.exists():
        shutil.rmtree(qdir)
    qdir.mkdir(parents=True)

    df = all_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])

    # ---- 全局日历 = 所有股票 date 的并集 ----
    global_cal = sorted(df["date"].unique())
    print(f"📅 全局日历长度 = {len(global_cal)}  "
          f"({global_cal[0].date()} ~ {global_cal[-1].date()})")

    # ---- calendars/day.txt ----
    (qdir / "calendars").mkdir()
    with open(qdir / "calendars" / f"{freq}.txt", "w", encoding="utf-8") as f:
        for d in global_cal:
            f.write(pd.Timestamp(d).strftime("%Y-%m-%d") + "\n")

    cal_to_idx = {pd.Timestamp(d): i for i, d in enumerate(global_cal)}

    # ---- 逐只股票：reindex 对齐 + 写 bin ----
    inst_lines = []
    for sym, g in df.groupby("symbol"):
        g = g.set_index("date").sort_index()
        aligned = g.reindex(global_cal)          # 缺失日期自动 NaN

        first = aligned.first_valid_index()
        if first is None:
            continue
        date_index = cal_to_idx[pd.Timestamp(first)]

        inst_lines.append((
            str(sym),
            pd.Timestamp(first).strftime("%Y-%m-%d"),
            pd.Timestamp(aligned.last_valid_index()).strftime("%Y-%m-%d"),
        ))

        fdir = qdir / "features" / str(sym).lower()
        fdir.mkdir(parents=True, exist_ok=True)

        for field in fields:
            if field not in aligned.columns:
                continue
            series = aligned[field].to_numpy(dtype=np.float32)
            bin_path = fdir / f"{field.lower()}.{freq}.bin"
            # ★★★ qlib bin 核心格式：[float32(date_index)] + [float32 × N] ★★★
            np.hstack([np.float32(date_index), series]).astype("<f").tofile(str(bin_path))

    # ---- instruments/all.txt ----
    (qdir / "instruments").mkdir()
    with open(qdir / "instruments" / f"{market}.txt", "w", encoding="utf-8") as f:
        for sym, s, e in inst_lines:
            f.write(f"{sym}\t{s}\t{e}\n")

    print(f"💾 股票数 = {len(inst_lines)}")
    print(f"✅ qlib 数据已写入: {qdir.resolve()}")
    return qdir


# ===========================================================================
# 【3 验证】用 qlib 自己的 D.features 读回来
# ===========================================================================
def verify_with_qlib(qlib_dir: str, freq: str = DEFAULT_FREQ):
    import qlib
    qlib.init(provider_uri={freq: str(qlib_dir)}, region="cn")
    from qlib.data import D

    insts = ["sh600000", "sh601318", "sz000001"]
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    df = D.features(insts, fields, start_time="2024-01-02", end_time="2024-05-31", freq=freq)
    print(f"\n📊 qlib D.features 读回 shape={df.shape}")
    print(df.head(6).to_string())
    return df


# ===========================================================================
# 【内置 demo】造两种 CSV + 转换 + 验证
# ===========================================================================
def run_demo():
    ROOT = Path("/workspace/qlib_demo_data")
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir()

    # ---- 造 3 支股票 120 天 ----
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-02", periods=120)
    stocks = {"sh600000": 10.0, "sh601318": 45.0, "sz000001": 12.0}
    all_frames = []
    for sym, base in stocks.items():
        rets = np.random.randn(len(dates)) * 0.02
        close = base * np.cumprod(1 + rets)
        open_ = close * (1 + np.random.randn(len(dates)) * 0.005)
        high = np.maximum(open_, close) * (1 + np.abs(np.random.randn(len(dates)) * 0.005))
        low = np.minimum(open_, close) * (1 - np.abs(np.random.randn(len(dates)) * 0.005))
        vol = np.random.randint(1_000_000, 10_000_000, len(dates)).astype(float)
        df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "symbol": sym,
            "open": open_.round(3), "high": high.round(3),
            "low": low.round(3),  "close": close.round(3),
            "volume": vol,
        })
        all_frames.append(df)
    big_all = pd.concat(all_frames, ignore_index=True)

    # 故意让 sz000001 少 3 行
    drop_dates = big_all.loc[(big_all["symbol"] == "sz000001")].iloc[[10, 50, 80]]["date"].tolist()
    big_all = big_all[~((big_all["symbol"] == "sz000001") & (big_all["date"].isin(drop_dates)))]

    # ---- 写两种 CSV ----
    # 模式 A：多文件
    dir_a = ROOT / "modeA_multi_csv"
    dir_a.mkdir()
    for sym, g in big_all.groupby("symbol"):
        g.to_csv(dir_a / f"{sym}.csv", index=False)

    # 模式 B：单文件
    file_b = ROOT / "modeB_single_big.csv"
    big_all.to_csv(file_b, index=False)

    # ---- 模式 A：多文件 ----
    print("\n" + "=" * 60)
    print("模式 A: 多文件  (--csv 目录)")
    print("=" * 60)
    q_a = to_qlib_bin(load_csv(str(dir_a)), str(ROOT / "qlib_bin_A"))

    # ---- 模式 B：单文件 ----
    print("\n" + "=" * 60)
    print("模式 B: 单文件  (--csv 文件)")
    print("=" * 60)
    q_b = to_qlib_bin(load_csv(str(file_b)), str(ROOT / "qlib_bin_B"))

    # ---- 验证 ----
    verify_with_qlib(q_a)
    verify_with_qlib(q_b)

    # ---- 打印一个 bin 的内部结构 ----
    bin_path = q_a / "features" / "sh600000" / "close.day.bin"
    raw = np.fromfile(str(bin_path), dtype="<f")
    print(f"\n🔍 手动读 {bin_path.relative_to(q_a)}：")
    print(f"   总 float32 = {len(raw)}  →  1 头 + {len(raw)-1} 数据")
    print(f"   date_index = {int(raw[0])}")
    print(f"   前 5 数据   = {raw[1:6]}")

    print("\n🎉 demo 跑完。真实使用：")
    print("   python3 csv_to_qlib_demo.py --csv 你的目录或.csv文件 --qlib 输出目录")


# ===========================================================================
# CLI 入口
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="CSV (单文件/多文件) → qlib 二进制",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 多文件 (目录下每支股票一个 csv)
  python3 csv_to_qlib_demo.py --csv /data/csvs/ --qlib /data/qlib_bin

  # 单文件 (一个大 csv，必须带 symbol 列)
  python3 csv_to_qlib_demo.py --csv /data/all.csv --qlib /data/qlib_bin

  # 内置 demo (自动造 mock 数据 + 跑两种模式 + 验证)
  python3 csv_to_qlib_demo.py --demo
        """,
    )
    ap.add_argument("--csv",  help="输入路径：要么是一个 .csv 文件，要么是含 .csv 的目录")
    ap.add_argument("--qlib", help="qlib 二进制输出目录 (将被创建/覆盖)")
    ap.add_argument("--freq", default=DEFAULT_FREQ, help="频率，默认 day")
    ap.add_argument("--market", default=DEFAULT_MARKET, help="instruments 文件名 (去掉 .txt)，默认 all")
    ap.add_argument("--demo", action="store_true", help="跑内置 demo (生成 mock csv + 两种模式 + D.features 验证)")
    args = ap.parse_args()

    if args.demo:
        run_demo()
        return

    if not args.csv or not args.qlib:
        ap.error("必须同时指定 --csv 和 --qlib，或用 --demo")

    df = load_csv(args.csv)
    to_qlib_bin(df, args.qlib, freq=args.freq, market=args.market)


if __name__ == "__main__":
    main()
