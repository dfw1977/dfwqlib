# -*- coding: utf-8 -*-
"""
CSV → Qlib 二进制数据 Demo
===========================

本脚本完整演示：
  ① 如何把多支股票的 CSV 日线数据，转换成 qlib 的二进制格式
  ② qlib 的三种存储（calendars / instruments / features）到底长什么样
  ③ 最后用 qlib 自己的 D.features() 读回来，验证格式正确

核心结论（qlib bin 格式，从 qlib/scripts/dump_bin.py 和
qlib/data/storage/file_storage.py 反推）：

  每个 features/<symbol>/<field>.day.bin 文件 =
      ┌──────────────────┬───────────────────────┐
      │ float32 date_index │ float32 × N 个数据点 │
      └──────────────────┴───────────────────────┘

  - date_index：这支股票在「全局日历」里第一个交易日的索引（0-based）
  - 后面 N 个 float32：按全局日历对齐后该字段的值，缺失日期填 NaN
  - 全部是 little-endian float32，用 numpy astype("<f").tofile 写盘

目录结构：
    qlib_data/
    ├── calendars/day.txt      # 每行一个日期 YYYY-MM-DD
    ├── instruments/all.txt    # symbol<TAB>start<TAB>end
    └── features/
        ├── sh600000/
        │   ├── open.day.bin
        │   ├── close.day.bin
        │   └── ...
        └── sz000001/
            └── ...
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. 路径
# ---------------------------------------------------------------------------
ROOT = Path("/workspace/qlib_demo_data")          # 整个 demo 的根目录
CSV_DIR = ROOT / "csv"                            # 模拟 CSV 放这
QLIB_DIR = ROOT / "qlib_bin"                      # qlib 二进制输出目录
FREQ = "day"                                      # 日线
MARKET = "all"                                    # instruments 的 market 名


# ===========================================================================
# 第一步：模拟生成 3 支股票的 CSV
# ===========================================================================
def make_mock_csvs():
    """造 3 支股票的模拟日线 CSV。"""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # 一个从 2024-01-02 开始的交易日序列（去掉周末，简单跳过即可）
    dates = pd.bdate_range("2024-01-02", periods=120)  # 120 个工作日

    symbols = {
        "sh600000": 10.0,   # 浦发银行风格
        "sh601318": 45.0,   # 中国平安风格
        "sz000001": 12.0,   # 平安银行风格
    }

    for sym, base_price in symbols.items():
        # 生成一个带一些随机游走的 OHLCV
        rets = np.random.randn(len(dates)) * 0.02
        close = base_price * np.cumprod(1 + rets)
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
        csv_path = CSV_DIR / f"{sym}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[CSV] 写入 {csv_path}  shape={df.shape}")

    # 故意让 sz000001 少几个交易日，演示 reindex 对齐的效果
    extra = pd.read_csv(CSV_DIR / "sz000001.csv")
    extra = extra.drop(index=[10, 50, 80])
    extra.to_csv(CSV_DIR / "sz000001.csv", index=False)
    print("[CSV] sz000001 故意删掉 3 行，模拟缺失交易日\n")


# ===========================================================================
# 第二步：CSV → qlib 二进制（手写实现，对应 dump_bin.py 的核心逻辑）
# ===========================================================================
CSV_FIELDS = ["open", "high", "low", "close", "volume"]   # 要转的字段


def csv_to_qlib_bin():
    """把 CSV_DIR 下所有 csv 转成 qlib 二进制。"""
    if QLIB_DIR.exists():
        shutil.rmtree(QLIB_DIR)

    # ---- 1) 读所有 CSV，汇总全局日历 ----
    all_dfs = []
    for csv_path in sorted(CSV_DIR.glob("*.csv")):
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.drop_duplicates("date").sort_values("date")
        all_dfs.append(df)
        print(f"读入 {csv_path.name}: {df.shape[0]} 行, "
              f"日期 {df['date'].min().date()} ~ {df['date'].max().date()}")

    all_df = pd.concat(all_dfs, ignore_index=True)
    global_calendar = sorted(all_df["date"].unique())
    print(f"\n全局日历长度 = {len(global_calendar)}")

    # ---- 2) 写 calendars/day.txt ----
    cal_dir = QLIB_DIR / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_txt = cal_dir / f"{FREQ}.txt"
    with open(cal_txt, "w", encoding="utf-8") as f:
        for d in global_calendar:
            f.write(pd.Timestamp(d).strftime("%Y-%m-%d") + "\n")
    print(f"[cal] 写入 {cal_txt}")

    # ---- 3) 为全局日历建一个 日期→索引 映射 ----
    cal_to_idx = {pd.Timestamp(d): i for i, d in enumerate(global_calendar)}

    # ---- 4) 逐只股票：对齐日历 + 写 features + 收集 instruments ----
    inst_lines = []     # (symbol, start, end)

    for sym, df in all_df.groupby("symbol"):
        df = df.set_index("date").sort_index()

        # 对齐到全局日历（缺失日期填 NaN）
        df_aligned = df.reindex(global_calendar)

        # date_index = 这支股票第一个有效交易日 在全局日历中的位置
        first_valid = df_aligned.first_valid_index()
        if first_valid is None:
            continue
        date_index = cal_to_idx[pd.Timestamp(first_valid)]

        start_dt = pd.Timestamp(first_valid).strftime("%Y-%m-%d")
        end_dt = pd.Timestamp(df_aligned.last_valid_index()).strftime("%Y-%m-%d")
        inst_lines.append((sym, start_dt, end_dt))
        print(f"\n[inst] {sym}: date_index={date_index}  "
              f"range=[{start_dt} ~ {end_dt}]")

        # 写每个 field 的 .bin
        feat_dir = QLIB_DIR / "features" / sym.lower()
        feat_dir.mkdir(parents=True, exist_ok=True)

        for field in CSV_FIELDS:
            series = df_aligned[field].to_numpy(dtype=np.float32)   # 会带 NaN
            bin_path = feat_dir / f"{field.lower()}.{FREQ}.bin"

            # ★ 核心：bin 格式 = [date_index 作为头部 1 个 float32] + [整个序列]
            blob = np.hstack([np.float32(date_index), series]).astype("<f")
            blob.tofile(str(bin_path))
            print(f"  -> {bin_path.name}  "
                  f"shape=({len(series)+1},)  "
                  f"date_index={date_index}  "
                  f"有效点数={np.isfinite(series).sum()}  "
                  f"NaN={np.isnan(series).sum()}")

    # ---- 5) 写 instruments/all.txt ----
    inst_dir = QLIB_DIR / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    inst_txt = inst_dir / f"{MARKET}.txt"
    with open(inst_txt, "w", encoding="utf-8") as f:
        for sym, start, end in inst_lines:
            f.write(f"{sym}\t{start}\t{end}\n")
    print(f"\n[inst] 写入 {inst_txt}")

    return global_calendar


# ===========================================================================
# 第三步：用 qlib 自己的 D.features 读回来验证
# ===========================================================================
def verify_with_qlib():
    """调用 qlib.data.D.features() 检查我们生成的数据是否可读。"""
    # 把刚才的输出目录注册给 qlib
    import qlib
    from qlib.config import C

    # 注意：provider_uri 可以是一个 dict，key 是 freq，value 是数据根目录
    provider_uri = {FREQ: str(QLIB_DIR)}
    qlib.init(provider_uri=provider_uri, region="cn")

    from qlib.data import D

    print("\n" + "=" * 60)
    print("用 qlib D.features() 读回验证")
    print("=" * 60)

    instruments = ["sh600000", "sh601318", "sz000001"]
    fields = ["$open", "$high", "$low", "$close", "$volume"]

    df = D.features(
        instruments,
        fields,
        start_time="2024-01-02",
        end_time="2024-05-31",
        freq=FREQ,
    )

    print(f"\nD.features 返回 shape = {df.shape}")
    print(df.head(12))

    # 抽样：sh600000 的 open 前 5 条（对比 CSV）
    csv_ref = pd.read_csv(CSV_DIR / "sh600000.csv", parse_dates=["date"])
    qlib_slice = df.loc[("sh600000", slice(None)), "$open"].head(5)
    csv_slice = csv_ref["open"].head(5).values
    print("\n核对 sh600000 的 open 前 5 个值：")
    print(f"  CSV 原始      : {csv_slice}")
    print(f"  Qlib D.features: {qlib_slice.values}")
    assert np.allclose(csv_slice, qlib_slice.values, equal_nan=True), "值对不上！"
    print("  ✅ 完全一致")

    # 抽样：sz000001 我们故意删了 3 天，看看 qlib 读出来是否也是 NaN
    csv_sz = pd.read_csv(CSV_DIR / "sz000001.csv", parse_dates=["date"])
    csv_sz = csv_sz.drop_duplicates("date").sort_values("date").set_index("date")
    qlib_sz_open = df.loc[("sz000001", slice(None)), "$open"]
    # qlib 返回的 index 里 multiindex 第二层是 Timestamp
    qlib_sz_open.index = qlib_sz_open.index.get_level_values(1)
    nan_count = qlib_sz_open.isna().sum()
    print(f"\nsz000001 在 qlib 里的 NaN 数量 = {nan_count}  （我们故意删了 3 行）")

    return df


# ===========================================================================
# 第四步（可选）：手写一个 mini reader，直接读 .bin，帮助理解
# ===========================================================================
def manual_read_bin_demo():
    """不依赖 qlib，手动读一个 .bin 文件，给你看头部 + 数据。"""
    bin_path = QLIB_DIR / "features" / "sh600000" / "close.day.bin"
    raw = np.fromfile(str(bin_path), dtype="<f")
    date_index = int(raw[0])
    data = raw[1:]
    print("\n[手动读 bin 演示]")
    print(f"  文件: {bin_path}")
    print(f"  总 float32 数: {len(raw)}  →  头部 1 个 + 数据 {len(data)} 个")
    print(f"  date_index = {date_index}  (这支股票第一个交易日在全局日历里的索引)")
    print(f"  前 5 个数据点 = {data[:5]}")


# ===========================================================================
# main
# ===========================================================================
if __name__ == "__main__":
    # 清理旧产物
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True, exist_ok=True)

    make_mock_csvs()
    csv_to_qlib_bin()

    # 打印生成的目录结构
    print("\n生成的目录结构：")
    for p in sorted(QLIB_DIR.rglob("*")):
        rel = p.relative_to(QLIB_DIR)
        if p.is_dir():
            print(f"  📁 {rel}/")
        else:
            print(f"    📄 {rel}  ({p.stat().st_size} bytes)")

    manual_read_bin_demo()
    verify_with_qlib()
    print("\n🎉 全部通过")
