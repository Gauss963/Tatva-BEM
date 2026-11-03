# 保留完整 CQ 歷史的短 lag 快取

2026-11-03：新增可選的 `tools/history_prefix_cache.py::cache_history_prefixes`。
原排程仍使用原本 policy；本頁沒有宣稱全程加速或 FEM 一致性。
實驗工具放在 package 之外，保留現有核及執行中事件的完整 package source 指紋。

## 為何準備這個改進

目前細網格權重採 Fortran-order `(22501, 531, 531)`，方便沿時間軸進行 FFT。
相對地，`weights[0]` 與少量 lag 的切片跨距很大。
`CompactRadauConvolution.append` 每步使用第二個 family 的 lag-zero 乘法，
小型直接卷積亦會反覆使用 lag 1–31。
執行中程序的一次 3 秒 OS 取樣看到 NumPy `DOUBLE_matmul`，
但這不足以定位 Python 行號、量化耗時比例或證明磁碟讀取就是主要瓶頸。

此 adapter 只將指定數目的完整 lag 矩陣複製到容量受限的 RAM prefix。
其餘所有 lag 仍由原檔案供應；跨越快取範圍的切片直接使用原來源。
不插值、不截斷歷史，也不更改既有直接／FFT 卷積演算法或時間網格。
transpose 共享同一份 prefix，不另複製一套歷史。

## 使用與記憶體

呼叫者須提供保持不變的 read-only float64 權重及相同 CQ plan：

```python
import runpy

cache_history_prefixes = runpy.run_path("bem/tools/history_prefix_cache.py")[
    "cache_history_prefixes"
]

cached, record = cache_history_prefixes(
    (first_history, second_history),
    capacity_bytes=144 * 1024**2,
    max_lags=32,
)
# 將 cached 的兩個 BoundaryHistory 傳給原 CompactRadauConvolution。
```

對目前兩個 531-DOF family，32 個 lag 的新增 array payload 為
**144364032 bytes／137.6763 MiB**，其中包含 lag zero。
這是 RAM 配額，不能解讀成磁碟實際只會讀取同樣大小的資料。
建 cache 時的來源 I/O 與複製成本也必須計入總時間。
`record` 列出實際 cached bytes／lags 與來源／prefix strides。
預算不足以容納一組完整 family lag 時不建立快取。

這筆 RAM **額外加在既有 kernel-FFT cache、工作陣列與求解器配額上**。
可選的 `simulate_prefix_cached_radau.py` 先從 `--max-memory-mb` 扣除指定 prefix 配額，
才呼叫原 planner。例如總 14336 MiB、prefix 144 MiB 時，原 planner 預算為 14192 MiB。
這仍是陣列估算，並非作業系統的 hard RSS cap。
adapter 禁止隱式轉成完整 NumPy array，須明確索引所需時間片。

## 可選的原求解器執行路徑

```bash
PYTHONPATH=bem/src python -B bem/tools/simulate_prefix_cached_radau.py KERNEL \
  --output NEW_RUN --history-prefix-cache-mb 144 --history-prefix-lags 32 \
  --max-memory-mb 14336 \
  --restart-seed bem/src/finiteblock_bem/resources/data/ev16_restart_float64.npz \
  --static-initialization fem-body-active-set --segment-substeps 64 128 \
  --contact-root-recovery --kinetic-root-seed --fft-cache-mb 8 --fft-rows 4
```

載入、初始平衡、時間推進與所有接觸／歷史數值函式保持原樣。
adapter 僅替換傳入原 `simulate_stages` 的兩個 `BoundaryHistory`，
在 private function binding 中使用原 `main`；不改寫其 module globals 或 kernel。
`history-prefix-policy.json` 另存 policy source hashes、原命令、配額及建 cache 耗時；
失敗時保留原 root 診斷與額外 policy 記錄，既有 output 不覆寫。
執行仍須使用與核相同的 runtime／x64 設定及外部 CPU 配額。

## 已驗證範圍與後續門檻

儲存 adapter 加上原 compact-history 測試共 **11 passed**：

- C／Fortran 磁碟來源、完整與跨 cache 邊界切片、只讀性、容量上限及檔案 SHA 未變動。
- 全部 130 個時間點的兩 family 原始／cached histories，包含直接與 streamed／cached FFT 路徑。
- 原展開 stage kernel 的獨立直接卷積對照，以及完整轉置的 dot-product identity。
- 比較門檻保留 `rtol=atol=2e-12`，沒有以 bitwise 相同推論不同記憶體 layout 的浮點加總。

加入 CLI 記憶體保留與原求解器參數轉接檢查後，本機共 **13 passed**。
另以 `check_prefix_cached_radau.py` 對同一已保存的完整 128 步 coarse kernel，
執行原 layout／cached layout 的 sliding 與初始化 JVP／VJP 對照；
控制保留原 `rtol=1e-7, atol=1e-18` 混合梯度門檻，結果尚待完成。

本機檢查使用額外平均 0.1 CPU quota，既有事件維持原 3.85 配額；
未暫停或改寫全程事件，81 個既有 frozen inputs 再次核對未變動。
線性歷史與 CLI 檢查尚不代表 cached policy 已通過
原始 45 ms 非線性軌跡、初始化 AD、梯度 refinement 或隔離總時間比較。
在這些控制完成前，保留[原版全程事件](full_original_radau_progress.md)作為基準。
[本機測試、CPU 配額、來源 hash 與基準程序快照](history_prefix_cache_results.json)
保存已完成的控制範圍。最初的 package 內版本移到 tools 後，已重跑相同 11 項檢查；
測試來源與目前公開來源逐檔相同，全部 26 個原 package 檔案集合及 hash 亦與完成核一致。
兩次本機控制皆保留，公開來源仍須通過對應 commit 的完整 CI。
