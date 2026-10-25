# 原始 FEM 四核心對照與界面輸出

2026-10-24。完整 BEM／FEM 等價與同精度速度優勢仍未達成。
本流程準備相同原始 FEM case 的本機參考，保留界面取樣頻率，避免為了比較而大量保存 bulk frames。
所有程式與輸出都在本 checkout；原 `/Volumes/Gauss-T7/tatva` 僅讀取。

2026-10-25修正：[前一次h2重跑錯把history儲存dtype當成模型dtype](reference_precision.md)，
該float32控制的79.5分鐘不列為原float64案例的效能基準。
正式float64短程的網格與第一筆位移／接觸狀態已與原檔逐位核對；完整動態仍待完成。

## 來源與設定

使用 Git revision `933ae1f3fe1c634391bdbd316e4efa5dbf0e6d5d` 的 archived FEM，
不是目前已演進的 root runner。每次將 source snapshot 與 `git archive` 的檔案 SHA-256 核對；
若快取內容或額外 Python module 不符合，會拒絕執行，不覆寫該快取。
`reference_manifest.json` 是執行前的來源／輸入／輸出設定記錄，完成與否以 `summary.json` 為準。
case、resolved RunConfig、worker 設定、archive／runner 指紋及 timing phases 都保存到新輸出資料夾。
case／solver取自`input/resolved.json`，並核對原TOML的指紋與source revision；
計算精度另外檢查TOML與resolved的一致性，不從statistics重建輸入。
`computational_dtype`描述計算模型，`stored_history_dtype`描述scalar history儲存。
既有 `simulation.h5` 或 `summary.json` 不會被覆寫。

原 2 mm case 不指定 `--mesh-size` 或 `--time-step`，沿用已保存的 mesh 與配置時間步。
`--mesh-size` 會另外投影 restart seed，僅用於明確的 FEM 網格研究。
本機 macOS 無法以此流程強制 CPU affinity；四核心是 thread 設定，須另外排除其他計算工作的干擾。

## 界面取樣與相位

原 FEM API 已支援獨立界面輸出。本 wrapper 現在可指定：

- `--normal-frames`／`--shear-frames`：bulk snapshots。
- `--interface-normal-frames`／`--interface-shear-frames`：獨立界面與 scalar history。

指定獨立取樣後，資料位於 `interface_high_rate`；舊資料則由 root `history` 與 `interface` 讀取。
比較工具從檔案本身讀取 column names，檢查完成筆數、有限值與來源未變更，
並以 `phase_id == 2` 選擇 shear。這可避免 normal-end 浮點時間戳捨入到 shear 起點時，
把 normal 階段的狀態納入成核指標。
省略獨立界面選項時，維持原本的輸出方式。

## 完整 45 ms 的儲存方式驗證

以已保存的 h=40 mm FEM 全事件作對照，維持四核心設定、物理參數、初始 seed、
自動穩定時間步、2 ms normal 與 45 ms shear：

|項目|原輸出|新輸出|
|---|---:|---:|
|bulk frames，含初始與 normal|9005|37|
|shear 界面／history 筆數|9000|9000|
|新高頻界面總筆數，含初始與 normal|—|9041|
|完整 HDF5 大小|35,361,659 bytes|5,819,953 bytes|

9000 筆 shear 的 17 個 scalar history 欄位、塑性滑移、累積滑移、摩擦強度與摩擦係數，
以及最後的 `u`、`v_half`、塑性／累積滑移皆逐位元相同。
這證明此儲存設定在該完整事件中沒有改變已比對的解；h40 是輸出回歸，
不能當成原 h2 FEM 精度或速度驗收。
新 run 的 solver call 約 48.5 秒，但當時有四-worker BEM 建核，故不作受控速度結論。
完整數字與來源見 [結果紀錄](fem_reference_results.json)。

## 執行原始 2 mm 參考

需完整 repository 的 archived Git object、Tatva/FEM 依賴與唯讀的原 case／restart 資料。
從 repository 根目錄執行：

```sh
FDBEM_CORES=4 bem/run.sh fem-reference --cores 4 \
  --normal-duration .002 --shear-duration .045 \
  --normal-frames 4 --shear-frames 32 \
  --interface-normal-frames 40 --interface-shear-frames 9000 \
  --output bem/outputs/fem-event16-authored-f64-h2-4c-interface

bem/run.sh compare COMPLETED_BEM_RUN \
  --fem-rerun bem/outputs/fem-event16-authored-f64-h2-4c-interface \
  --report-dir NEW_REPORT
```

`--fem-rerun` 與舊選項 `--coarse-fem` 相容；它保留原 archived h2 曲線，另加本機 FEM 曲線。
不指定 mesh-size，因此上面的正式參考沒有改成粗網格。
正式原h2／float64完整動態尚未完成，不能把短程初始化或不同精度的控制算成完成。
`bem/outputs/fem2mm-after-bem-20261024.json`記錄的是已排除的float32控制。
wrapper 的 timings 不含監督程序的 polling 延遲，並將 preparation／simulation／output 範圍列出。
同精度比較仍需 BEM 誤差達標，並另核對冷建核、初始化、JIT、推進與輸出成本。

完成後使用`tools/compare_fem_reference.py`對原始HDF5核對，不插值不同step的軌跡。
它要求完整h2／float64、相同來源／輸入與精確相同的shear step IDs，
再逐項比對全部scalar history、四個界面狀態、初始場及最後u／v_half／滑移。
預設rtol=1e-6、atol=1e-9，另保留逐位相同、最大絕對差與relative L2資訊；
超出門檻會保存失敗報告並以exit1結束。這是FEM來源驗收，並不代表BEM已重現。

```sh
FDBEM_CORES=1 PYTHONPATH=bem/src python -B bem/tools/compare_fem_reference.py \
  bem/outputs/fem-event16-authored-f64-h2-4c-interface \
  --output bem/outputs/fem-event16-authored-f64-field-comparison
```
