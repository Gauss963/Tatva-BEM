# 可重跑的完整 Radau AD 驗證

2026-10-28。`tools/check_compact_radau_ad.py` 將
[含初始化的完整 RK adjoint](radau_history_validation.md) 整理成命令列入口。
舊 `check_radau_ad.py` 的 bounded direct-scan reference 保持原樣；兩者測試均保留。

## 執行

```sh
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 FDBEM_CORES=4 \
  JAX_ENABLE_X64=true python -B bem/tools/check_compact_radau_ad.py \
  /path/to/completed-radau-run \
  --operator-audit /path/to/passed-original-rk-audit \
  --output /path/to/fresh-ad-control \
  --jvp-batch-size 3 --fft-cache-mb 8 --fft-rows 2 --max-memory-mb 12288
```

預設微分 selected healed static active set，包含初始位移、塑性偏移及實測力。
`--initialization-gradient fixed` 則明確固定初態，作條件式控制。
兩種模式都要求原始非線性 root tolerance，以及同一完整 stages 已通過的
original-RK causal audit；核對時間網格、stage SHA256、kernel、float64 權重、
實測初始場與 restart seed。若原 seed 路徑不在新環境，可用同 SHA256 的
隨套件原始 float64 seed，或明確指定 `--restart-seed`；不能改成另一批數值。

`--preflight-only` 不計算梯度；短程 prefix 必須明確給 `--allow-prefix`。
進度、記憶體估算、來源指紋與最終梯度結果分開保存。
既有 stages 改動後，不能沿用原 causal-audit 的通過結果。

## 高效使用 AD 與記憶體

一次 reverse pass 同時得到五個接觸參數與每個 stage actuator 的 cotangent。
獨立 JVP 檢查涵蓋五個參數基底、sinusoidal loading 與固定亂數 loading 方向；
default 分成 3+3+1，跨同一批方向與兩個 stage 共用每族 kernel FFT。
每步微分平衡根與局部摩擦 map，沒有記錄每次 Newton 迭代的計算圖。
驗證器在 passes 之間清除 JIT caches，這段成本納入總時間。

保守估算包含兩族 pending／input 陣列、批次 FFT padding、streamed kernel rows、
saved stages、dense root workspace 及 2 GiB runtime reserve；**不是 process RSS 硬上限**，
mapped 權重頁面與 OS caches 另計。對目前 531 retained DOFs：

|45 ms 時間網格|JVP batch|AD 陣列峰值估算|12 GiB 陣列門檻|
|---|---:|---:|---|
|2 µs|3|9879 MiB|通過|
|2 µs|7|17061 MiB|拒絕|
|1 µs|1|9971 MiB|通過|
|1 µs|3|17152 MiB|拒絕|

這些是記憶體規劃，尚未執行 fine AD。CLI 另檢查 static-body phase 的預算，
超出時在進入昂貴計算前拒絕，並提示縮小 batch 或 FFT rows。

## 已完成的控制

[實際 128 步雙 block 控制](radau_ad_cli_results.json) 的 initialized 與 fixed
兩種模式均通過。initialized 的最大相對非零方向差約 **6.93e-13**；
與先前實際控制相比，五個 reverse gradients 及全部 actuator cotangents 逐位元相同。
這裡檢查相對尺度與 1e-18 的 absolute floor，避免只用大尺度絕對容差掩蓋
normal／tangential penalty 的小梯度。

測試另從真實小型兩有限體、fresh complete RK kernel 開始，經過 healed body
初始化、實際滑移、完整原始 RK audit，最後跑 initialized AD CLI。
並檢查 preflight 不產生梯度、低記憶體預算會拒絕，以及 stages 被改動後拒用舊證明。

這些尚未證明 fine／45 ms 梯度收斂或 FEM 效能優勢。
停止載入事件時間、材料、幾何與 bulk operator 的導數仍不在此驗證範圍；
recorded stage actuator 作為獨立輸入。原始 Tatva 維持唯讀。
