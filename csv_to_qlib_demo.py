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
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import numpy as np

# 让 qlib 官方脚本能被 import
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "scripts"))
from dump_bin import DumpDataAll, DumpDataFix, DumpDataUpdate  # noqa: E402


DEFAULT_FREQ = "day"


# ===========================================================================
# 【Tushare 拉数模块】 — 社区 investment_data 项目的轻量重写
# ===========================================================================
# Tushare API 清单（按优先级）:
#   pro.daily()       — 日线行情: ts_code, trade_date, open, high, low, close, vol, amount
#   pro.adj_factor()  — 复权因子: ts_code, trade_date, adj_factor
#   pro.trade_cal()   — 交易日历: exchange, is_open, cal_date
#   pro.stock_basic() — 股票存续期: ts_code, list_date, delist_date, list_status
#   pro.index_daily() — 指数日线（可选）
#
# 字段映射（Tushare → Qlib）:
#   trade_date (YYYYMMDD)  →  date (YYYY-MM-DD)
#   ts_code (000001.SZ)    →  symbol (SZ000001)      代码翻转
#   vol                    →  volume                   量纲转换
#   adj_factor × close     →  adjclose                 用于算复权
#   adjclose / close       →  factor                   Qlib 用它
#   open/high/low/close    →  open/high/low/close
#
# 复权逻辑（继承 YahooNormalizeCN1d）:
#   factor = adjclose / close
#   OHLC   = OHLC × factor      （前复权价）
#   volume = volume / factor    （复权后量）
#   首日 close 归一化（manual_adj_data）
# ===========================================================================

# Qlib 代码格式: Tushare "000001.SZ" → Qlib "SZ000001"
def ts_code_to_qlib(ts_code: str) -> str:
    """Tushare 代码 → Qlib 代码"""
    code, market = ts_code.split(".")  # "000001", "SZ"
    return f"{market}{code}"


def _get_tushare_pro(token: str | None = None):
    """获取 Tushare pro_api 实例"""
    import tushare as ts
    tok = token or os.environ.get("TUSHARE")
    if not tok:
        raise RuntimeError("需要 Tushare Token: 设 --tushare-token 参数 或 export TUSHARE=xxx")
    ts.set_token(tok)
    return ts.pro_api()


def fetch_tushare_daily(
    symbols: list[str],
    start_date: str,
    end_date: str,
    token: str | None = None,
    sleep: float = 0.3,
) -> pd.DataFrame:
    """
    从 Tushare pro_api 拉日线行情 + 复权因子，返回合并后的 DataFrame。

    等价于 investment_data/tushare/dump_a_stock_eod_price.py 的 get_daily()
    但按 symbol 循环（更轻量，适合小批量）。

    Parameters
    ----------
    symbols : list[str]
        Tushare 格式代码列表, 如 ["000001.SZ", "600000.SH"]
    start_date : str
        "YYYYMMDD" 格式, 如 "20230101"
    end_date : str
        "YYYYMMDD" 格式, 如 "20241231"
    token : str, optional
        Tushare token, 默认读环境变量 TUSHARE
    sleep : float
        每次 API 调用间隔秒数（避免频控）, 默认 0.3

    Returns
    -------
    pd.DataFrame
        列: ts_code, trade_date, open, high, low, close, vol, amount, adj_factor, adjclose
    """
    pro = _get_tushare_pro(token)
    all_frames = []

    for i, ts_code in enumerate(symbols):
        print(f"  [{i+1}/{len(symbols)}] 拉 {ts_code} ...", end=" ")
        for attempt in range(3):
            try:
                price = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                adj = pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"失败: {e}")
                    price = adj = pd.DataFrame()
                else:
                    time.sleep(2)

        if price.empty or adj.empty:
            print("空 (已跳过)")
            time.sleep(sleep)
            continue

        df = pd.merge(price, adj, on=["ts_code", "trade_date"], how="inner")
        df["adjclose"] = df["close"] * df["adj_factor"]
        all_frames.append(df)
        print(f"{len(df)} 行 ✅")
        time.sleep(sleep)

    if not all_frames:
        raise RuntimeError("Tushare 拉取全部为空")
    return pd.concat(all_frames, ignore_index=True)


def fetch_tushare_by_date(
    start_date: str,
    end_date: str,
    token: str | None = None,
    sleep: float = 0.3,
) -> pd.DataFrame:
    """
    按交易日拉取全市场（investment_data 的方式, 适合全量/日更）。
    如果你的积分够高，推荐用这个（更高效）。
    """
    pro = _get_tushare_pro(token)

    # 先拿交易日历
    cal = pro.trade_cal(exchange="SSE", is_open="1", start_date=start_date, end_date=end_date)
    trade_dates = cal["cal_date"].sort_values().tolist()
    print(f"📅 {start_date} ~ {end_date} 共 {len(trade_dates)} 个交易日")

    all_frames = []
    for i, td in enumerate(trade_dates):
        print(f"  [{i+1}/{len(trade_dates)}] {td} ...", end=" ")
        for attempt in range(3):
            try:
                price = pro.daily(trade_date=td)
                adj = pro.adj_factor(trade_date=td)
                break
            except Exception as e:
                if attempt == 2:
                    print(f"失败: {e}")
                    price = adj = pd.DataFrame()
                else:
                    time.sleep(2)

        if price.empty or adj.empty:
            print("空")
            time.sleep(sleep)
            continue

        df = pd.merge(price, adj, on="ts_code", how="inner")
        df["adjclose"] = df["close"] * df["adj_factor"]
        all_frames.append(df)
        print(f"{len(df)} 只 ✅")
        time.sleep(sleep)

    if not all_frames:
        raise RuntimeError("Tushare 按日拉取全部为空")
    return pd.concat(all_frames, ignore_index=True)


# --- Qlib 格式标准化 ---
def normalize_to_qlib_csv(
    df: pd.DataFrame,
    output_dir: Path,
    fields: str = "date,open,high,low,close,volume,factor,amount",
    manual_adj: bool = True,
) -> set:
    """
    把 Tushare 原始 DataFrame 标准化成 Qlib 多文件 CSV 格式。

    复用社区 investment_data 的思路 + YahooNormalizeCN1d 的复权逻辑:
      1. factor = adjclose / close
      2. OHLC = OHLC × factor, volume = volume / factor
      3. _manual_adj_data: 按首日 close 归一化（Qlib 离线因子训练需要）

    Parameters
    ----------
    df : pd.DataFrame
        Tushare 原始数据, 需含 ts_code, trade_date, open, high, low, close,
        vol, amount, adj_factor, adjclose
    output_dir : Path
        输出目录 (多文件模式, 文件名 = qlib_symbol.csv, 无 symbol 列)
    fields : str
        要输出的字段
    manual_adj : bool
        是否做首日 close 归一化 (默认 True, 与官方 Normalize 一致)

    Returns
    -------
    set[str]
        生成的 qlib_symbol 集合
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    field_list = [f.strip() for f in fields.split(",")]

    qlib_symbols = set()

    for ts_code, group in df.groupby("ts_code"):
        # 字段映射
        out = group.rename(columns={
            "trade_date": "date",
            "vol": "volume",
        })[["date", "open", "high", "low", "close", "volume", "amount",
            "adj_factor", "adjclose"]].copy()

        out["date"] = pd.to_datetime(out["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
        out = out.sort_values("date").reset_index(drop=True)

        # --- 复权逻辑 (YahooNormalizeCN1d) ---
        # factor = adjclose / close, ffill 填充
        out["factor"] = (out["adjclose"] / out["close"]).replace([np.inf, -np.inf], np.nan).ffill()
        out["factor"] = out["factor"].fillna(1.0)

        # OHLC × factor (前复权)
        for col in ["open", "high", "low", "close"]:
            out[col] = out[col] * out["factor"]
        # volume / factor (复权量)
        out["volume"] = out["volume"] / out["factor"]

        # 首日 close 归一化 (manual_adj_data)
        if manual_adj and len(out) > 0:
            first_close = out["close"].iloc[0]
            if first_close > 0:
                for col in ["open", "high", "low", "close"]:
                    out[col] = out[col] / first_close
                out["volume"] = out["volume"] * first_close

        # 选取输出字段
        out = out[field_list]

        qlib_symbol = ts_code_to_qlib(ts_code)
        out.to_csv(output_dir / f"{qlib_symbol}.csv", index=False)
        qlib_symbols.add(qlib_symbol)

    return qlib_symbols


# --- 包装函数 ---
def tushare_to_qlib(
    qlib_dir: str,
    start_date: str,
    end_date: str,
    symbols: str | None = None,
    symbols_file: str | None = None,
    token: str | None = None,
    by_date: bool = False,
    mode: str = "auto",
    max_workers: int = 1,
):
    """
    【一键】 Tushare → CSV → Qlib bin

    全流程:
      ① Tushare pro_api 拉 daily + adj_factor
      ② 标准化成 Qlib 多文件 CSV
      ③ 调用官方 DumpDataAll / DumpDataUpdate 转 .bin
      ④ 打印产物统计

    Parameters
    ----------
    qlib_dir : str
        Qlib bin 输出目录
    start_date / end_date : str
        "YYYYMMDD"
    symbols : str, optional
        Tushare 代码, 逗号分隔. 例: "000001.SZ,600000.SH,600519.SH"
    symbols_file : str, optional
        股票列表文件 (每行一个 ts_code)
    token : str, optional
        Tushare token
    by_date : bool
        True = 按交易日全市场拉 (需要较高积分)
        False = 按 symbol 循环拉 (默认)
    mode : str
        auto | all | update | fix
    max_workers : int
        dump_bin 并行数
    """
    print("\n" + "=" * 60)
    print("📈 Tushare → Qlib 全流程")
    print("=" * 60)

    # 1. 解析 symbol 列表
    syms: list[str] | None = None
    if by_date:
        print("🧮 模式: 按交易日拉全市场 (by_date=True)")
    elif symbols:
        syms = [s.strip() for s in symbols.split(",") if s.strip()]
        print(f"🎯 指定 {len(syms)} 只股票: {syms}")
    elif symbols_file:
        syms = [line.strip() for line in open(symbols_file) if line.strip()]
        print(f"📃 从文件加载 {len(syms)} 只: {symbols_file}")
    else:
        raise ValueError("必须指定 --symbols, --symbols-file, 或 --by-date")

    print(f"📅 日期范围: {start_date} ~ {end_date}")

    # 2. 拉数据
    print("\n① Tushare 拉数据 ...")
    if by_date:
        df = fetch_tushare_by_date(start_date, end_date, token)
    else:
        assert syms is not None
        df = fetch_tushare_daily(syms, start_date, end_date, token)
    print(f"   ✅ 拉取 {len(df)} 行, {df['ts_code'].nunique()} 只")

    # 3. 标准化成 Qlib CSV
    with tempfile.TemporaryDirectory(prefix="tushare_csv_") as tmp:
        csv_dir = Path(tmp) / "qlib_source"
        print(f"\n② 标准化成 Qlib 多文件 CSV → {csv_dir}")
        qsyms = normalize_to_qlib_csv(df, csv_dir)
        print(f"   ✅ 生成 {len(qsyms)} 只股票 CSV")

        # 4. 转 bin (复用已有的 run)
        print(f"\n③ 调用官方 dump_bin → {qlib_dir}")
        run(mode=mode, csv_path=str(csv_dir), qlib_dir=qlib_dir,
            freq=DEFAULT_FREQ, max_workers=max_workers)

    # 5. 打印产物统计
    qdir = Path(qlib_dir)
    features_dir = qdir / "features"
    n_syms = len(list(features_dir.iterdir())) if features_dir.exists() else 0
    cal_count = len((qdir / "calendars" / f"{DEFAULT_FREQ}.txt").read_text().splitlines()) if (qdir / "calendars" / f"{DEFAULT_FREQ}.txt").exists() else 0
    print(f"\n🎉 完成！")
    print(f"   📂 {qdir.resolve()}")
    print(f"   📈 {n_syms} 只股票 × {cal_count} 个交易日")


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
# 【Tushare demo】— 用 mock 数据验证全流程
# ===========================================================================
def run_tushare_demo():
    """不用真实 Tushare token, 用 mock 数据跑通全链路"""
    import numpy as np
    ROOT = Path("/workspace/qlib_demo_data")
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir()

    print("\n" + "=" * 60)
    print("📈 Tushare → Qlib Demo (mock 数据, 无 token)")
    print("=" * 60)

    # 1. 生成 mock Tushare 格式数据
    np.random.seed(42)
    dates = pd.date_range("2023-01-03", "2024-12-31", freq="B")
    symbols = ["000001.SZ", "600000.SH", "000333.SZ", "600519.SH", "000858.SZ"]

    print(f"\n① 生成 mock Tushare 数据: {len(symbols)} 只 × {len(dates)} 天")
    all_frames = []
    for ts_code in symbols:
        n = len(dates)
        base = np.random.uniform(8, 1800)
        rets = np.random.randn(n) * 0.02
        close = base * np.cumprod(1 + rets)
        adj_factor = np.cumprod(1 + np.random.randn(n) * 0.002)  # 模拟少量复权

        df = pd.DataFrame({
            "ts_code": ts_code,
            "trade_date": dates.strftime("%Y%m%d"),
            "open": (close + np.random.randn(n) * 0.1).round(3),
            "high": (close + abs(np.random.randn(n) * 0.2)).round(3),
            "low":  (close - abs(np.random.randn(n) * 0.2)).round(3),
            "close": close.round(3),
            "vol": np.random.randint(10000, 1000000, n),
            "amount": np.random.randint(1000000, 100000000, n),
            "adj_factor": adj_factor.round(6),
        })
        df["adjclose"] = (df["close"] * df["adj_factor"]).round(3)
        all_frames.append(df)

    mock = pd.concat(all_frames, ignore_index=True)
    print(f"   ✅ {len(mock)} 行")

    # 2. 标准化
    csv_dir = ROOT / "tushare_csv"
    print(f"\n② normalize_to_qlib_csv → {csv_dir}")
    qsyms = normalize_to_qlib_csv(mock, csv_dir)
    print(f"   ✅ {len(qsyms)} 只: {sorted(qsyms)}")

    # 3. dump bin
    qdir = ROOT / "qlib_bin_from_tushare"
    print(f"\n③ dump_bin → {qdir}")
    run(mode="auto", csv_path=str(csv_dir), qlib_dir=str(qdir),
        freq=DEFAULT_FREQ, max_workers=1)

    # 4. 验证
    print("\n④ 二进制验证")
    import numpy as np2
    cal_lines = (qdir / "calendars" / "day.txt").read_text().splitlines()
    inst_lines = (qdir / "instruments" / "all.txt").read_text().splitlines()
    print(f"   📅 日历: {len(cal_lines)} 天 ({cal_lines[0]} ~ {cal_lines[-1]})")
    print(f"   📈 股票: {len(inst_lines)} 只")
    for line in inst_lines:
        print(f"      {line}")

    # 读一只的 bin
    sym = list(qsyms)[0].lower()
    open_bin = qdir / "features" / sym / "open.day.bin"
    data = np2.fromfile(open_bin, dtype="float32")
    print(f"\n   📊 {sym}/open.day.bin: len={len(data)}, date_index={int(data[0])}")
    print(f"      前5个值: {data[1:6]}")
    print(f"      数据一致性: {'✅' if len(data) == len(cal_lines) + 1 else '❌'} "
          f"(expect {len(cal_lines)+1})")

    print("\n🎉 Tushare demo 完成！")


# ===========================================================================
# CLI 入口
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="CSV → Qlib 二进制（全量/增量 直接调用官方 DumpDataAll/DumpDataUpdate）\n"
                    "也支持 Tushare 一键拉数：--tushare 系列参数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
## 模式 1: CSV → Qlib (原有)
  python3 csv_to_qlib_demo.py --csv /data/csvs/  --qlib /data/qlib_bin
  python3 csv_to_qlib_demo.py --csv /data/today.csv --qlib /data/qlib_bin

## 模式 2: Tushare → Qlib (新增)
  python3 csv_to_qlib_demo.py --qlib ~/.qlib/qlib_data/cn_data \\
      --tushare-symbols "000001.SZ,600000.SH,600519.SH" \\
      --tushare-start 20230101 --tushare-end 20241231 \\
      --tushare-token $TUSHARE

  python3 csv_to_qlib_demo.py --qlib ~/.qlib/qlib_data/cn_data \\
      --tushare-symbol-file /path/to/ts_codes.txt \\
      --tushare-start 20240101 --tushare-end today

## 模式 3: Demo (mock 数据, 无 token)
  python3 csv_to_qlib_demo.py --demo            # CSV demo
  python3 csv_to_qlib_demo.py --tushare-demo    # Tushare mock demo

## 底层用到的 Tushare API
  pro.daily()       — 日线行情
  pro.adj_factor()  — 复权因子
  pro.trade_cal()   — 交易日历 (by-date 模式)
  pro.stock_basic() — 股票列表 (可加)
""",
    )
    # --- 原有 CSV 参数 ---
    ap.add_argument("--csv",      help="输入 csv 文件或目录")
    ap.add_argument("--qlib",     help="qlib 二进制输出目录")
    ap.add_argument("--freq",     default=DEFAULT_FREQ)
    ap.add_argument("--mode",     default="auto", choices=["auto", "all", "update", "fix"])
    ap.add_argument("--workers",  type=int, default=2)
    ap.add_argument("--demo",     action="store_true",
                    help="原有 CSV demo (mock 数据)")

    # --- 新增 Tushare 参数 ---
    ap.add_argument("--tushare-symbols",    help="股票代码, 逗号分隔. 例: '000001.SZ,600000.SH'")
    ap.add_argument("--tushare-symbol-file",help="股票列表文件 (每行一个 ts_code)")
    ap.add_argument("--tushare-start",      help="起始日期 YYYYMMDD")
    ap.add_argument("--tushare-end",        help="结束日期 YYYYMMDD")
    ap.add_argument("--tushare-token",      help="Tushare token (默认读环境变量 TUSHARE)")
    ap.add_argument("--tushare-by-date",    action="store_true",
                    help="按交易日拉全市场 (需要较高积分)")
    ap.add_argument("--tushare-demo",        action="store_true",
                    help="Tushare mock demo (不需要 token)")

    args = ap.parse_args()

    # --- Tushare demo ---
    if args.tushare_demo:
        run_tushare_demo()
        return

    # --- 原有 CSV demo ---
    if args.demo:
        run_demo()
        return

    # --- Tushare 模式 ---
    if args.tushare_symbols or args.tushare_symbol_file or args.tushare_by_date:
        if not args.qlib:
            ap.error("Tushare 模式需要 --qlib 指定输出目录")
        if not args.tushare_start or not args.tushare_end:
            ap.error("Tushare 模式需要 --tushare-start 和 --tushare-end (YYYYMMDD)")

        tushare_to_qlib(
            qlib_dir=args.qlib,
            start_date=args.tushare_start,
            end_date=args.tushare_end,
            symbols=args.tushare_symbols,
            symbols_file=args.tushare_symbol_file,
            token=args.tushare_token,
            by_date=args.tushare_by_date,
            mode=args.mode,
            max_workers=args.workers,
        )
        return

    # --- 原有 CSV 模式 ---
    if not args.csv or not args.qlib:
        ap.error("必须同时指定 --csv 和 --qlib, 或用 --demo, 或用 --tushare-*")

    run(args.mode, args.csv, args.qlib, args.freq, args.workers)


if __name__ == "__main__":
    main()
