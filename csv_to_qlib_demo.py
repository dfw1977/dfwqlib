# -*- coding: utf-8 -*-
"""
CSV → Qlib 二进制 转换脚本（CLI，支持增量更新）
==================================================

用法
----
# ① 全量初始化（qlib_bin 不存在时自动走；或 --mode all 强制）
python3 csv_to_qlib_demo.py --csv /data/csv_dir --qlib /data/qlib_bin

# ② 增量更新（qlib_bin 已存在时自动走；或 --mode update 强制）
python3 csv_to_qlib_demo.py --csv /data/new_daily.csv --qlib /data/qlib_bin

# ③ 内置 demo：造两种 CSV → 全量初始化 → 追加新数据 → 增量更新 → 验证
python3 csv_to_qlib_demo.py --demo

模式识别
--------
  qlib_dir/ 存在  → 自动进入增量模式 (update)
  qlib_dir/ 不存在 → 自动进入全量模式 (all)
  --mode all|update 可以强制覆盖

增量模式做了什么
----------------
  1) 读老 calendars/day.txt    → old_cal
  2) 读老 features/*.bin       → 把每只股票每个 field 的老数据拿回来
  3) new_cal = old_cal ∪ 新数据的日期（全局日历自动变长）
  4) 每只股票：
     - 新股票：按 new_cal reindex 直接写（和全量一样）
     - 老股票：merge(老数据, 新数据) → 按 new_cal reindex → 重写整个 bin
               （date_index 可能因 new_cal 扩展而改变）
  5) 写新的 calendars / instruments

为什么重写整个 bin 而不是 append
---------------------------------
  方案 A（官方 DumpDataUpdate append）：
    只追加新日期，要算 start_index/end_index，gap 填 NaN，逻辑复杂；
    中间某天漏了补不上。
  方案 B（本脚本 reindex + 重写）：
    只要能读到老 bin 里的数据，reindex(new_cal) 自动对齐，
    加新日期、补旧数据、股票停牌/复牌都能搞定；
    bin 通常几百 KB，重写一次 I/O 成本可以忽略。
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 默认常量
# ---------------------------------------------------------------------------
DEFAULT_FREQ = "day"
DEFAULT_MARKET = "all"
DEFAULT_FIELDS = ["open", "high", "low", "close", "volume"]


# ===========================================================================
# 【工具函数】读老 qlib 数据（增量模式专用）
# ===========================================================================
def _read_calendars(qlib_dir: Path, freq: str) -> list:
    """读 qlib calendars/day.txt，返回 pd.Timestamp 列表。"""
    p = qlib_dir / "calendars" / f"{freq}.txt"
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [pd.Timestamp(line.strip()) for line in f if line.strip()]


def _read_bin(qdir: Path, symbol: str, field: str, freq: str) -> tuple:
    """
    读一支股票一个 field 的老 bin。
    返回 (date_index: int, series: np.ndarray)
    series 长度 = 旧日历长度，index 从 date_index 开始。
    文件不存在返回 (None, None)。
    """
    bin_path = qdir / "features" / symbol.lower() / f"{field.lower()}.{freq}.bin"
    if not bin_path.exists():
        return None, None
    raw = np.fromfile(str(bin_path), dtype="<f")
    date_index = int(raw[0])
    series = raw[1:]
    return date_index, series


def _read_all_old_features(qdir: Path, old_cal: list, freq: str, fields: list) -> dict:
    """
    把 qdir 下所有老 bin 读出来。
    返回 dict[symbol][field] = pd.Series(index=old_cal)
    """
    result = {}
    features_root = qdir / "features"
    if not features_root.exists():
        return result

    cal_to_ts = list(old_cal)  # 按原顺序

    for sym_dir in features_root.iterdir():
        if not sym_dir.is_dir():
            continue
        sym = sym_dir.name
        result[sym] = {}
        for field in fields:
            date_index, series = _read_bin(qdir, sym, field, freq)
            if series is None:
                continue
            # 老 series 对应的真实日历区间: [date_index, date_index + len(series))
            idx = cal_to_ts[date_index : date_index + len(series)]
            result[sym][field] = pd.Series(series, index=idx, name=field)
    return result


# ===========================================================================
# 【1 读 CSV】自动识别单文件 / 多文件
# ===========================================================================
def load_csv(data_path: str) -> pd.DataFrame:
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"--csv 路径不存在: {data_path}")

    if p.is_file():
        print(f"📄 单文件模式: {p}")
        df = pd.read_csv(p)
        if "symbol" not in df.columns:
            raise ValueError(f"单文件模式必须有 symbol 列，当前列: {list(df.columns)}")
    elif p.is_dir():
        csvs = sorted(list(p.glob("*.csv")) + list(p.glob("*.CSV")))
        if not csvs:
            raise FileNotFoundError(f"目录下没有 csv: {p}")
        print(f"📁 多文件模式: {p}  ({len(csvs)} 个 csv)")
        frames = []
        for fp in csvs:
            d = pd.read_csv(fp)
            sym_from_fname = fp.stem
            if "symbol" not in d.columns:
                d["symbol"] = sym_from_fname
            else:
                d["symbol"] = d["symbol"].fillna(sym_from_fname)
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
    else:
        raise ValueError(f"--csv 既不是文件也不是目录: {data_path}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
    print(f"   读入 shape={df.shape}  symbols={df['symbol'].nunique()}  "
          f"日期 {df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


# ===========================================================================
# 【2 转 qlib】全量 or 增量
# ===========================================================================
def to_qlib_bin(all_df: pd.DataFrame, qlib_dir: str,
                freq: str = DEFAULT_FREQ, market: str = DEFAULT_MARKET,
                fields=None, mode: str = "auto"):
    """
    mode: auto | all | update
      auto    → qlib_dir 不存在 = all, 存在 = update
      all     → 强制全量（先 rm qdir 再重建）
      update  → 强制增量（qdir 不存在会报错）
    """
    fields = fields or DEFAULT_FIELDS
    qdir = Path(qlib_dir)

    if mode == "auto":
        mode = "update" if qdir.exists() else "all"

    # ---------- 全量模式 ----------
    if mode == "all":
        if qdir.exists():
            shutil.rmtree(qdir)
        qdir.mkdir(parents=True)
        print(f"\n🆕 全量初始化  mode=all  → {qdir.resolve()}")
        _dump(all_df, qdir, freq, market, fields, old_cal=None, old_features={})
        return qdir

    # ---------- 增量模式 ----------
    if not qdir.exists():
        raise FileNotFoundError(f"--mode update 但 qlib_dir 不存在: {qdir}")
    old_cal = _read_calendars(qdir, freq)
    old_features = _read_all_old_features(qdir, old_cal, freq, fields)
    print(f"\n🔄 增量更新  mode=update  → {qdir.resolve()}")
    print(f"   老日历长度={len(old_cal)}  老股票数={len(old_features)}")

    # 用新数据里的所有日期 ∪ 老日历 → 新日历
    new_dates = sorted(pd.to_datetime(all_df["date"].unique()))
    new_cal = sorted(set(old_cal) | set(new_dates))
    print(f"   新日期数={len(new_dates)}  合并后新日历长度={len(new_cal)}")

    _dump(all_df, qdir, freq, market, fields, old_cal=old_cal, old_features=old_features)
    return qdir


def _dump(all_df: pd.DataFrame, qdir: Path, freq: str, market: str,
          fields: list, old_cal=None, old_features=None):
    """
    核心：不管全量还是增量，最后都走到这里。
      - 全量：old_features = {}
      - 增量：old_features 里放着每只股票每个 field 的老数据 Series(index=old_cal)
    """
    old_cal = old_cal or []
    old_features = old_features or {}

    df = all_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])

    # 全局日历 = 老日历 ∪ 新数据日期（并集，自动处理新交易日）
    if old_cal:
        global_cal = sorted(set(old_cal) | set(pd.to_datetime(df["date"].unique())))
    else:
        global_cal = sorted(df["date"].unique())
    print(f"📅 全局日历长度={len(global_cal)}  "
          f"({pd.Timestamp(global_cal[0]).date()} ~ {pd.Timestamp(global_cal[-1]).date()})")

    # calendars/day.txt
    (qdir / "calendars").mkdir(parents=True, exist_ok=True)
    with open(qdir / "calendars" / f"{freq}.txt", "w", encoding="utf-8") as f:
        for d in global_cal:
            f.write(pd.Timestamp(d).strftime("%Y-%m-%d") + "\n")

    cal_to_idx = {pd.Timestamp(d): i for i, d in enumerate(global_cal)}

    # 逐支股票
    inst_lines = []
    new_symbols = set(df["symbol"].unique())
    old_symbols = set(old_features.keys())
    all_symbols = sorted(old_symbols | new_symbols)

    for sym in all_symbols:
        # 把老数据先拼成一个 DataFrame（每个 field 一列）
        old_frames = []
        if sym in old_features:
            old_frames.append(pd.concat(old_features[sym].values(), axis=1))
            # 如果老 bin 里缺某些 field，用列名补齐
            for f in fields:
                if f not in old_frames[-1].columns:
                    old_frames[-1][f] = np.nan

        # 新数据
        new_frames = []
        if sym in new_symbols:
            new_rows = df[df["symbol"] == sym].set_index("date")
            new_frames.append(new_rows[fields])

        frames = old_frames + new_frames
        if not frames:
            continue

        # 按行拼接：老数据 (date, field) 在上，新数据在下
        merged = pd.concat(frames, axis=0)
        # 同一个 (date, field) 可能重复（新数据覆盖老数据），取最后一个
        merged = merged[~merged.index.duplicated(keep="last")]

        # ★ 按全局日历对齐，缺失自动 NaN（这一步自动处理增量）
        aligned = merged.reindex(global_cal)

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
            # ★★★ qlib bin = [float32(date_index)] + [float32 × N 数据] ★★★
            np.hstack([np.float32(date_index), series]).astype("<f").tofile(str(bin_path))

        # 增量日志
        tag = "🆕新" if sym not in old_symbols else "🔄更新"
        if sym not in old_symbols:
            print(f"   {tag} {sym}  date_index={date_index}")
        else:
            old_end_sym = pd.Timestamp(old_features[sym].get(next(iter(old_features[sym])), pd.Series(dtype=float)).index.max()).date()
            new_end = pd.Timestamp(aligned.last_valid_index()).date()
            delta = (new_end - old_end_sym).days
            print(f"   {tag} {sym}  date_index={date_index}  新增≈{delta}天")

    # instruments/all.txt  — symbol 用大写（与官方 dump_bin.py 输出一致）
    (qdir / "instruments").mkdir(parents=True, exist_ok=True)
    with open(qdir / "instruments" / f"{market}.txt", "w", encoding="utf-8") as f:
        for sym, s, e in inst_lines:
            f.write(f"{sym.upper()}\t{s}\t{e}\n")

    print(f"💾 股票总数={len(inst_lines)}")
    print(f"✅ qlib 数据已写入: {qdir.resolve()}")


# ===========================================================================
# 【3 验证】用 qlib D.features()
# ===========================================================================
def verify_with_qlib(qlib_dir: str, freq: str = DEFAULT_FREQ,
                     start="2024-01-02", end="2024-07-15",
                     insts=None, fields=None):
    import qlib
    qlib.init(provider_uri={freq: str(qlib_dir)}, region="cn")
    from qlib.data import D
    insts = insts or ["sh600000", "sh601318", "sz000001"]
    fields = fields or ["$open", "$close", "$volume"]
    df = D.features(insts, fields, start_time=start, end_time=end, freq=freq)
    print(f"\n📊 qlib D.features  shape={df.shape}")
    print(df.groupby(level=0).count().to_string())  # 每只股票有效行数
    print(df.head(6).to_string())
    return df


# ===========================================================================
# 【内置 demo】全量 → 增量 → 验证
# ===========================================================================
def run_demo():
    ROOT = Path("/workspace/qlib_demo_data")
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir()

    # ---- 造 3 支股票 60 天 (2024-01-02 ~ 03-22) 作为第一批 ----
    np.random.seed(42)
    dates_a = pd.bdate_range("2024-01-02", periods=60)
    stocks = {"sh600000": 10.0, "sh601318": 45.0, "sz000001": 12.0}
    def _fake_ohlcv(dates, base, seed):
        rng = np.random.RandomState(seed)
        rets = rng.randn(len(dates)) * 0.02
        close = base * np.cumprod(1 + rets)
        open_ = close * (1 + rng.randn(len(dates)) * 0.005)
        high = np.maximum(open_, close) * (1 + np.abs(rng.randn(len(dates)) * 0.005))
        low  = np.minimum(open_, close) * (1 - np.abs(rng.randn(len(dates)) * 0.005))
        vol  = rng.randint(1_000_000, 10_000_000, len(dates)).astype(float)
        return pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "symbol": [f"s{seed}" for _ in dates],  # 先不填，后面补
            "open": open_.round(3), "high": high.round(3),
            "low": low.round(3), "close": close.round(3),
            "volume": vol,
        })

    frames_a = []
    for i, (sym, base) in enumerate(stocks.items()):
        d = _fake_ohlcv(dates_a, base, i)
        d["symbol"] = sym
        frames_a.append(d)
    batch_a = pd.concat(frames_a, ignore_index=True)
    csv_a = ROOT / "batchA_initial.csv"
    batch_a.to_csv(csv_a, index=False)

    # ---- 造增量批次：75 天 (04-01 ~ 07-12)，含老股票 + 1 支新股票 sh600519 ----
    dates_b = pd.bdate_range("2024-04-01", periods=75)
    frames_b = []
    for i, (sym, base) in enumerate({**stocks, "sh600519": 1700.0}.items()):
        d = _fake_ohlcv(dates_b, base, i + 100)
        d["symbol"] = sym
        frames_b.append(d)
    batch_b = pd.concat(frames_b, ignore_index=True)
    csv_b = ROOT / "batchB_incremental.csv"
    batch_b.to_csv(csv_b, index=False)

    # ────────────────────────────────
    # Step 1: 全量初始化（只喂 batch_a）
    # ────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1: 全量初始化  输入 batchA_initial.csv (60 天, 3 支股票)")
    print("=" * 60)
    qdir = ROOT / "qlib_bin"
    to_qlib_bin(pd.read_csv(csv_a), qdir, mode="auto")
    verify_with_qlib(qdir, start="2024-01-02", end="2024-03-25",
                     insts=["sh600000", "sh601318", "sz000001"])

    # ────────────────────────────────
    # Step 2: 增量更新（喂 batch_b，日历从 60 扩到 135，新增 sh600519）
    # ────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: 增量更新  输入 batchB_incremental.csv (75 天, 含 1 支新股票 sh600519)")
    print("=" * 60)
    to_qlib_bin(pd.read_csv(csv_b), qdir, mode="auto")   # qdir 已存在 → 自动 update
    verify_with_qlib(qdir, start="2024-01-02", end="2024-07-15",
                     insts=["sh600000", "sh601318", "sz000001", "sh600519"])

    # ────────────────────────────────
    # Step 3: 手动读 bin 验证新旧数据都在
    # ────────────────────────────────
    print("\n🔍 手动读 bin 验证增量结果：")

    # sh600000 旧股票，应该有 135 个 float32
    raw = np.fromfile(str(qdir / "features" / "sh600000" / "close.day.bin"), dtype="<f")
    print(f"   sh600000/close.day.bin: 总 float32={len(raw)}  "
          f"date_index={int(raw[0])}  数据={len(raw)-1}个  "
          f"前5={raw[1:6]}  后5={raw[-5:]}")

    # sh600519 新股票，应该只有从 04-01 开始的有效数据
    raw2 = np.fromfile(str(qdir / "features" / "sh600519" / "close.day.bin"), dtype="<f")
    print(f"   sh600519/close.day.bin: 总 float32={len(raw2)}  "
          f"date_index={int(raw2[0])}  数据={len(raw2)-1}个  "
          f"前5={raw2[1:6]}")

    # 验证 bin 大小一致（3 支老股票应该都是 136 = 1 + 135）
    for sym in ["sh600000", "sh601318", "sz000001"]:
        size = (qdir / "features" / sym / "close.day.bin").stat().st_size
        print(f"   {sym}/close.day.bin  size={size} bytes  "
              f"(期望 136 × 4 = 544 bytes)")

    print("\n🎉 demo 完成。")
    print("   真实使用示例：")
    print("   python3 csv_to_qlib_demo.py --csv /data/daily.csv --qlib /data/qlib_bin")


# ===========================================================================
# CLI 入口
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="CSV → qlib 二进制（全量 / 增量自动识别）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
模式识别:
  qlib_bin/ 不存在 → 自动全量  (rmdir + rebuild)
  qlib_bin/ 存在   → 自动增量  (读老 bin → merge 新数据 → reindex → 重写)
  --mode all|update 可强制覆盖

示例:
  # 全量初始化（第一次用）
  python3 csv_to_qlib_demo.py --csv /data/csvs/ --qlib /data/qlib_bin

  # 增量更新（每天跑一次）
  python3 csv_to_qlib_demo.py --csv /data/today.csv --qlib /data/qlib_bin

  # 强制全量重灌
  python3 csv_to_qlib_demo.py --csv /data/full.csv --qlib /data/qlib_bin --mode all

  # 强制增量（调试用，qlib_bin 必须已存在）
  python3 csv_to_qlib_demo.py --csv /data/new.csv --qlib /data/qlib_bin --mode update

  # 内置 demo (自动跑全量 + 增量 + D.features 验证)
  python3 csv_to_qlib_demo.py --demo
        """,
    )
    ap.add_argument("--csv",    help="输入：要么是 .csv 文件（单文件模式），要么是含 .csv 的目录（多文件模式）")
    ap.add_argument("--qlib",   help="qlib 二进制输出目录")
    ap.add_argument("--freq",   default=DEFAULT_FREQ,  help="频率，默认 day")
    ap.add_argument("--market", default=DEFAULT_MARKET, help="instruments 文件名(去掉 .txt)，默认 all")
    ap.add_argument("--mode",   default="auto", choices=["auto", "all", "update"],
                    help="all=全量  update=增量  auto=自动（默认）")
    ap.add_argument("--demo",   action="store_true",   help="跑内置 demo")
    args = ap.parse_args()

    if args.demo:
        run_demo()
        return
    if not args.csv or not args.qlib:
        ap.error("必须同时指定 --csv 和 --qlib，或用 --demo")

    df = load_csv(args.csv)
    to_qlib_bin(df, args.qlib, freq=args.freq, market=args.market, mode=args.mode)


if __name__ == "__main__":
    main()
