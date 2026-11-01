# 複數線性求解與未使用回復矩陣的成本

新增可選 `prepare_trace_policy_radau.py`，結合既有 JAX raw 組裝、明確 general LU
與 impedance-only 的惰性回復計算。原 Galerkin 積分、支承、接觸律、時間步與所有因果項不變。
完整細網格 event-16 的 FEM 一致性與同精度速度／穩定性優勢仍未完成。
正在執行的原版建核沒有換路徑、遷移或重啟。

## 由實際程序確認成本

2026-11-01 對原 PID 97341 做 3 s／10 ms 的 macOS stack sampling。
主執行緒主要等待 ThreadPool，兩個 worker 都在矩陣運算；
原 `raw_layer_sampling.emit` 本來就會 `pool.map(output, ...)`，因此不需要新增同一層平行化。
取樣同時看到 `ZSYTRS`、`ZGETRS` 與 GEMM，不能把這段視為完整執行時間的百分比分解。

本機 SciPy 1.17.1 的 `solve` 預設 `assume_a=None`，會檢查矩陣結構；
明確 `assume_a='gen'` 使用一般 LU，保留完整複數矩陣與 bilinear transposes。
[SciPy 官方文件](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html)
亦說明自動結構辨識及 general／symmetric 對應的 LAPACK 路徑。
這是求解實作的選擇，沒有改成 Hermitian 問題、加正則化或事後對稱化。

原 `TraceSchurTransfer.condense` 即使只要求阻抗，仍配置 `recover` 並計算完整 reduced response。
惰性版本僅移除三個未使用的 recovery assignments 與最後的 recovery product，
阻抗所需的原運算保留。要求完整位移回復時，仍走原完整公式。
小模型及本次全部細網格抽樣的 lazy-only 阻抗均要求**逐位元一致**。

## 相同原始矩陣的兩 worker 配對計時

原細網格 `h=10, contact_h=2, end/corner=0.015625, Q12`，531 history DOF，
45 ms／2 µs 的 M=65536 contour。每個頻率只做一次原版 Q12 組裝，保存全部 V/H/D/mass，
四種策略共用同一批原始 bytes；三輪計時反轉策略順序，取 median。
每筆時間是**兩個頻率、兩個 worker 的阻抗加 work compliance 批次**，不含原始矩陣組裝。

|策略|branch 0，indices 0／655|branch 1，indices 11000／11001|
|---|---:|---:|
|原 automatic solve／完整 recovery|1.3449 s|1.6368 s|
|general LU|1.1207 s|1.0867 s|
|僅略過未使用 recovery|1.3056 s|1.5865 s|
|general LU + lazy recovery|1.0390 s|1.0334 s|

合併策略的時間比為 **1.294／1.584**，分別減少約 **22.7%／36.9%** 的這段求解時間。
單獨刪除未使用回復只有約 3% 改善，不能從程式外觀推定它是主要成本。
branch 1 這一對位於目前長程建核的插值區間內，但四個頻率仍不是整段頻譜的統計保證。

相同 raw matrices 在所有策略前後 SHA 不變。
各 repeat 的 impedance／work compliance 另驗證 Frobenius 與 infinity norms，
最大相對差約 **6.61e-13**，原門檻為 `1e-8`。
完整 physical boundary reconstruction 另外獨立計算與核對，四種策略全通過。
這些額外稽核時間不混入上表。

兩批原始組裝分別花 **104.759 s／60.377 s**，完整 benchmark 子程序 **236.489 s**，
包含保存 raw inputs 與所有稽核。不能拿上表當成完整建核時間，也不能把本結果
與前次 JAX raw 組裝比直接相乘，宣稱整體加速。

## 可選建核器與完整短程閉環

`trace_solve_policy.py` 使用私有 function globals，general 路徑作用於 V 的多 RHS solve、
Trace-Schur elimination 與 work compliance。沒有修改原 module globals、prepared operators 或原 Tatva。
`prepare_trace_policy_radau.py` 重用既有 durable JAX adapter，只替換私有 transfer／work bindings。
每次建核另記錄 policy、來源與 runtime；JAX-only context 不能靜默接手此 kernel，反之亦然。

直接 lag-zero 的兩個頻率綁定到**未修改的原 transfer**，繼續使用原 SciPy layers 與求解，
避免之前已發現的小 stage 元素精度問題。
獨立原 body-compliance audit 仍解回原 prepared geometry；sampler、接受門檻、FFT、
全部 lags、shift wrap、startup 及 signal／journal cleanup 均沿用。
原額外 JAX workspace allowance 保留，沒有以它宣稱 hard RSS cap。

十二項測試通過，包含四項 policy controls、C／Fortran 完整 16-step kernel 對照獨立 dense
original CQ，及六項保存資料檢查。另新建完整 128-step candidate kernel，再直接使用其
保存權重，對照原版 kernel 的完整滑動與初始化 AD：

|項目|相對差|
|---|---:|
|全部 causal weights|2.05e-14|
|全部保存 stage 位移|1.09e-13|
|全部接觸 response|2.23e-13|
|五個接觸參數梯度|3.39e-12|
|全部 stage loading gradient|1.08e-13|

原版與 candidate 都達到最大累積滑移約 **1.033821980 mm**。
七方向 JVP／VJP 及跨 backend 梯度維持 `rtol=1e-7, atol=1e-18`，最大 normalized error
約 **0.001704 < 1**。起始阻抗另要求逐位元相同。
此控制仍是 h=160、2 µs × 128、診斷載入 7812.5 mm/s、8／16 摩擦子步的粗網格案例；
不是原 20 mm/s 的完整 event-16，沒有材料／幾何或內生停止時刻導數。

完整 candidate 建核子程序 **21.163 s**、滑動／AD 控制 **12.101 s**，包含各自啟動成本。
它們沒有配對的獨立冷啟動完整原版計時，因此不宣稱完整 kernel 的加速比。

```sh
PYTHONPATH=bem/src JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 \
  python -B bem/tools/prepare_trace_policy_radau.py case-or-mesh.json \
  --output /path/to/new-kernel --dt 2e-6 --duration .000256 \
  --workers 1 --durable-build
```

上例的 mesh 與載入由輸入決定；完整原事件需另做資源與收斂檢查。
此路徑尚未套用到正在執行的細網格 cache。
所有數值控制用平均 3.85 CPU quota，結束恢復相同原三個程序，已付費 prefix SHA 前後不變。
本輪三個控制的外層成本共 **300.329 s**，全部保留在原 BEM wall time。
完整紀錄與成本見 [結果 JSON](trace_solve_policy_results.json)；
原 `/Volumes/Gauss-T7/tatva` 維持唯讀，原全程 FEM／BEM 比較門檻不變。
