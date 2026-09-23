# -*- coding: utf-8 -*-
"""
CSV → Qlib 二进制数据 Demo
===========================

本脚本完整演示：
  ① CSV 两种格式 → qlib 二进制
  ② qlib 的三种存储（calendars / instruments / features）到底长什么样
  ③ 用 qlib 自己的 D.features() 读回来验证

CSV 两种输入格式（核心区别：数据在"多个文件"还是"一个文件里"）：
┌─────────────────────────────────────────────────────────────────────┐
│ 模式 A: 目录模式 —— 每支股票一个 CSV 文件                           │
│   csv/                                                              │
│   ├── sh600000.csv   ← 文件名就是股票代码                           │
│   ├── sh601318.csv                                                 │
│   └── sz000001.csv                                                 │
│   每个文件: date, open, high, low, close, volume  (可含 symbol 列) │
│                                                                     │
│ 模式 B: 单文件模式 —— 一个 CSV 里所有股票，靠 symbol 列区分         │
│   csv/all_stocks.csv                                                │
│   列: date, symbol, open, high, low, close, volume                  │
│   3000 支股票可压在一个文件里                                        │
└─────────────────────────────────────────────────────────────────────┘

qlib bin 格式（从 dump_bin.py / file_storage.py 反推）：
  features/<symbol>/<field>.day.bin =
      ┌───────────────────┬───────────────────────────────┐
      │ float32 date_index │ float32 × N  数据点（缺失填NaN）│
      └───────────────────┴───────────────────────────────┘
  date_index = 这支股票第一个交易日 在「全局日历」中的索引（0-based）
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. 路径
# ---------------------------------------------------------------------------
ROOT = Path("/workspace/qlib_demo_data")
CSV_DIR = ROOT / "csv"                # 两种模式的 CSV 都会生成到这里
QLIB_DIR_A = ROOT / "qlib_bin_modeA"  # 模式 A 输出
QLIB_DIR_B = ROOT / "qlib_bin_modeB"  # 模式 B 输出
FREQ = "day"
MARKET = "all"
CSV_FIELDS = ["open", "high", "low", "close", "volume"]


# ===========================================================================
# 模拟数据：两种 CSV 格式，源数据是同一套（方便对比验证）
# ===========================================================================
def _gen_raw_stocks():
    """内部：生成 3 支股票 120 天的 OHLCV dict。"""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-02", periods=120)
    symbols = {
        "sh600000": 10.0,
        "sh601318": 45.0,
        "sz000001": 12.0,
    }
    result = {}
    for sym, base in symbols.items():
        rets = np.random.randn(len(dates)) * 0.02
        close = base * np.cumprod(1 + rets)
        open_ = close * (1 + np.random.randn(len(dates)) * 0.005)
        high = np.maximum(open_, close) * (1 + np.abs(np.random.randn(len(dates)) * 0.005))
        low = np.minimum(open_, close) * (1 - np.abs(np.random.randn(len(dates)) * 0.005))
        volume = np.random.randint(1_000_000, 10_000_000, len(dates)).astype(float)
        df = pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "symbol": sym,
            "open": open_.round(3),
            "high": high.round(3),
            "low": low.round(3),
            "close": close.round(3),
            "volume": volume,
        })
        result[sym] = df
    return result


def make_mock_csvs():
    """同时生成两种格式的 CSV。"""
    if CSV_DIR.exists():
        shutil.rmtree(CSV_DIR)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    stocks = _gen_raw_stocks()

    # ---- 模式 A：每支股票一个 csv ----
    mode_a_dir = CSV_DIR / "modeA_one_per_stock"
    mode_a_dir.mkdir(exist_ok=True)
    for sym, df in stocks.items():
        path = mode_a_dir / f"{sym}.csv"
        df.to_csv(path, index=False)
        print(f"[模式A] {path.name}  shape={df.shape}")

    # 故意让 sz000001 少 3 行，演示 reindex 对齐
    df_sz = pd.read_csv(mode_a_dir / "sz000001.csv")
    df_sz = df_sz.drop(index=[10, 50, 80])
    df_sz.to_csv(mode_a_dir / "sz000001.csv", index=False)
    print(f"[模式A] sz000001 故意删 3 行  -> shape={df_sz.shape}")

    # ---- 模式 B：一个大 csv 所有股票 ----
    all_df = pd.concat(stocks.values(), ignore_index=True)
    # 同样故意让 sz000001 少 3 行（用 date 对齐删掉那 3 天）
    drop_dates = stocks["sz000001"].iloc[[10, 50, 80]]["date"].tolist()
    all_df = all_df[~((all_df["symbol"] == "sz000001") & (all_df["date"].isin(drop_dates)))]
    mode_b_path = CSV_DIR / "modeB_all_in_one.csv"
    all_df.to_csv(mode_b_path, index=False)
    print(f"\n[模式B] {mode_b_path.name}  shape={all_df.shape}  (包含所有 3 支股票, "
          f"其中 sz000001 故意少 3 行)")


# ===========================================================================
# 核心：通用的「DataFrame → qlib 二进制」转换
# ===========================================================================
def _df_to_qlib_bin(all_df: pd.DataFrame, qlib_dir: Path):
    """
    输入一个「已经读好的 DataFrame」（必须含 symbol / date 列），
    输出 qlib 标准目录结构到 qlib_dir。

    不管这个 DataFrame 是来自模式 A（多文件 concat）还是模式 B（单文件），
    后续处理完全一模一样 —— 这就是两种模式真正的唯一区别：读数据的方式。
    """
    qlib_dir.mkdir(parents=True, exist_ok=True)
    all_df = all_df.copy()
    all_df["date"] = pd.to_datetime(all_df["date"])

    # 同一支股票可能有重复 date（多文件拼接时），去重
    all_df = all_df.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])

    # ---- 1) 全局日历：所有股票的 date 并集 ----
    global_calendar = sorted(all_df["date"].unique())
    print(f"  全局日历长度 = {len(global_calendar)}")

    # ---- 2) calendars/day.txt ----
    cal_dir = qlib_dir / "calendars"
    cal_dir.mkdir(exist_ok=True)
    with open(cal_dir / f"{FREQ}.txt", "w", encoding="utf-8") as f:
        for d in global_calendar:
            f.write(pd.Timestamp(d).strftime("%Y-%m-%d") + "\n")

    cal_to_idx = {pd.Timestamp(d): i for i, d in enumerate(global_calendar)}

    # ---- 3) 逐只股票：对齐日历 + 写 features + 收集 instruments ----
    inst_lines = []

    for sym, df in all_df.groupby("symbol"):
        df = df.set_index("date").sort_index()
        df_aligned = df.reindex(global_calendar)   # 缺失日期自动 NaN 填充

        first_valid = df_aligned.first_valid_index()
        if first_valid is None:
            continue
        date_index = cal_to_idx[pd.Timestamp(first_valid)]

        inst_lines.append((
            sym,
            pd.Timestamp(first_valid).strftime("%Y-%m-%d"),
            pd.Timestamp(df_aligned.last_valid_index()).strftime("%Y-%m-%d"),
        ))

        feat_dir = qlib_dir / "features" / sym.lower()
        feat_dir.mkdir(parents=True, exist_ok=True)

        for field in CSV_FIELDS:
            series = df_aligned[field].to_numpy(dtype=np.float32)
            bin_path = feat_dir / f"{field.lower()}.{FREQ}.bin"
            # ★ 核心：bin = [float32(date_index)] + [float32 × N 数据点]
            np.hstack([np.float32(date_index), series]).astype("<f").tofile(str(bin_path))

    # ---- 4) instruments/all.txt ----
    inst_dir = qlib_dir / "instruments"
    inst_dir.mkdir(exist_ok=True)
    with open(inst_dir / f"{MARKET}.txt", "w", encoding="utf-8") as f:
        for sym, start, end in inst_lines:
            f.write(f"{sym}\t{start}\t{end}\n")

    print(f"  股票数 = {len(inst_lines)}")
    return global_calendar


# ===========================================================================
# 模式 A：目录 → 多文件逐支读取 → concat → 转二进制
# ===========================================================================
def modeA_dir_to_qlib():
    data_path = CSV_DIR / "modeA_one_per_stock"
    print(f"\n{'='*60}")
    print(f"模式 A: data_path = 目录 {data_path}")
    print(f"{'='*60}")

    if QLIB_DIR_A.exists():
        shutil.rmtree(QLIB_DIR_A)

    # 逐文件读，文件名就是股票代码
    dfs = []
    for csv_path in sorted(data_path.glob("*.csv")):
        df = pd.read_csv(csv_path)
        # 如果文件里没有 symbol 列，就用文件名 stem 补
        if "symbol" not in df.columns:
            df["symbol"] = csv_path.stem
        print(f"  读 {csv_path.name}  shape={df.shape}")
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    _df_to_qlib_bin(all_df, QLIB_DIR_A)


# ===========================================================================
# 模式 B：单个大 CSV 文件 → 转二进制
# ===========================================================================
def modeB_single_file_to_qlib():
    data_path = CSV_DIR / "modeB_all_in_one.csv"
    print(f"\n{'='*60}")
    print(f"模式 B: data_path = 单个文件 {data_path}")
    print(f"{'='*60}")

    if QLIB_DIR_B.exists():
        shutil.rmtree(QLIB_DIR_B)

    all_df = pd.read_csv(data_path)
    print(f"  读入大 CSV  shape={all_df.shape}  symbols={all_df['symbol'].nunique()}")
    _df_to_qlib_bin(all_df, QLIB_DIR_B)


# ===========================================================================
# 用 qlib D.features() 验证
# ===========================================================================
def verify_with_qlib(qlib_dir: Path, label: str):
    import qlib
    qlib.init(provider_uri={FREQ: str(qlib_dir)}, region="cn")

    from qlib.data import D
    instruments = ["sh600000", "sh601318", "sz000001"]
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    df = D.features(instruments, fields, start_time="2024-01-02", end_time="2024-05-31", freq=FREQ)

    print(f"\n✅ [{label}] D.features shape={df.shape}")
    print(df.head(5).to_string())

    # 抽样校验
    qlib_slice = df.loc[("sh600000", slice(None)), "$open"].head(5).values
    print(f"  sh600000 $open 前 5 = {qlib_slice}")

    # sz000001 NaN 数量（我们故意删了 3 行）
    nan_cnt = df.loc[("sz000001", slice(None)), "$open"].isna().sum()
    print(f"  sz000001 NaN 数 = {nan_cnt}  （预期 3）")
    assert nan_cnt == 3, f"sz000001 NaN 应该是 3，实际是 {nan_cnt}"

    return df


# ===========================================================================
# 打印 bin 文件结构，辅助理解
# ===========================================================================
def dump_bin_structure(qlib_dir: Path, label: str):
    print(f"\n📁 [{label}] 目录结构：")
    for p in sorted(qlib_dir.rglob("*")):
        rel = p.relative_to(qlib_dir)
        if p.is_dir():
            print(f"   📂 {rel}/")
        else:
            print(f"     📄 {rel}  ({p.stat().st_size} bytes)")

    # 读一个 bin 展示内部
    bin_path = qlib_dir / "features" / "sh600000" / "close.day.bin"
    raw = np.fromfile(str(bin_path), dtype="<f")
    print(f"\n🔍 [{label}] 手动读 {bin_path.relative_to(qlib_dir)}：")
    print(f"   总 float32 = {len(raw)}  →  1 个头部 + {len(raw)-1} 个数据点")
    print(f"   date_index = {int(raw[0])}")
    print(f"   前 5 个数据 = {raw[1:6]}")


# ===========================================================================
# main
# ===========================================================================
if __name__ == "__main__":
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)

    # 1. 生成两种 CSV
    make_mock_csvs()

    # 2. 模式 A
    modeA_dir_to_qlib()
    dump_bin_structure(QLIB_DIR_A, "模式A")
    verify_with_qlib(QLIB_DIR_A, "模式A")

    # 3. 模式 B
    modeB_single_file_to_qlib()
    dump_bin_structure(QLIB_DIR_B, "模式B")
    verify_with_qlib(QLIB_DIR_B, "模式B")

    print("\n🎉 两种模式全部通过！")
