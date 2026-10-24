# 逐點驗證的頻譜儲存壓縮

2026-10-24。此實驗降低頻譜快取的磁碟需求，保留完整時間窗與非零歷史。
它不改摩擦律，也不改善既有 BEM 與 FEM 的物理解差距。
[原始數值、來源與 SHA-256](polynomial_spectrum_results.json) 保留驗證範圍。

## 方法與誤差範圍

對已完成的離散半頻譜，逐區間以 16 次 Chebyshev 多項式儲存矩陣。
在**每一個原始頻率格點**比較重建值與來源矩陣，而非只查插值節點：

```text
||M_rebuilt[k] - M_source[k]||_F <= 1e-18 + 1e-13*||M_source[k]||_F.
```

不符合條件的區間細分；短區間直接保存原值。
重建以 Clenshaw 求值，原始與重建頻譜均採標準 CQ 的實軸端點投影。
不完整資料、非有限來源值、來源變更或超出儲存預算時，不寫入完成標記。
讀取會核對每個檔案的 checksum 與連續覆蓋，並支援 IFFT 所需的分批列讀取。

這個界只相對於**已計算的離散矩陣**。來源 BEM 的自適應頻率插值仍是經有限點
檢查的方法；此實驗沒有把它升格為全頻率或連續算子的誤差定理。
工作座標 `D*L*D`、原 CQ contour 與其轉回 physical L 的方式均維持不變。

## 完整粗網格 4 us／45 ms 的結果

來源為 71 retained DOFs、16385 個頻率點、5500 個修正 lags、11250 個時間步。

|項目|結果|
|---|---:|
|原頻譜陣列資料量|1.322 GB|
|壓縮後陣列檔案|0.309 GB|
|頻譜儲存比例|約 4.27 倍縮減|
|直接保存／多項式表示格點|3136／13249|
|區間數|139|
|最大正規化格點誤差|0.899，門檻為 1|

這是**頻譜**的縮減；約 222 MB 的修正時間權重仍完整保留，沒有宣稱總記憶體也減少 4.27 倍。

再由壓縮頻譜重新建立時間權重與完整因果彈性啟動，重新求解完整 45 ms：

- 時間權重最大絕對差約 4.99e-18，啟動位移差約 1.46e-12 mm。
- 對未壓縮 L 軌跡，滑移最大差約 1.09e-8 mm，剪力差約 2.77e-6 MPa；
  actuator 相同，28.896 ms 的 coverage hold 相同。
- 代回原完整 Z 歷史後，平衡 residual 加初始誤差界約 1.70e-8 MPa，低於原 1e-7 MPa 容差。
- 對原 Z 五參數 adjoint，最大相對差約 2.24e-6；相對未壓縮 L 的額外差約 7.79e-7。
  精確滑動、開閉、弱化 clip 與等值分支變更皆為零。

這是固定 bulk、lift、記錄載入與 active branches 下的條件式 AD。
它不包含材料／幾何、初始化或 hold 事件時間的導數。
微小頻譜差仍能放大為較大的非線性軌跡差，故不以小矩陣差取代完整解與梯度核對。

## 使用與未完成項目

```sh
FDBEM_CORES=1 PYTHONPATH=bem/src python -B bem/tools/polynomial_spectrum.py \
  COMPLETED_KERNEL --output NEW_SPECTRUM
FDBEM_CORES=1 PYTHONPATH=bem/src python -B bem/tools/rebuild_polynomial_kernel.py \
  NEW_SPECTRUM --output NEW_KERNEL
FDBEM_CORES=1 bem/run.sh resolvent-replay ORIGINAL_RUN \
  --prepared-kernel NEW_KERNEL --output NEW_REPLAY
PYTHONPATH=bem/src python -B bem/tools/audit_replay_residual.py ORIGINAL_RUN NEW_REPLAY
PYTHONPATH=bem/src python -B bem/tools/adjoint_resolvent.py ORIGINAL_RUN NEW_REPLAY \
  --reference-gradient ORIGINAL_GRADIENT_JSON
```

依 `bem/run.sh` 設定 BLAS limits。所有輸出需使用新資料夾；來源頻譜不會被覆寫。
本次壓縮程序約 7 秒，未重新執行 BEM transfer；核重建另記錄時間。
來源冷建立含其他診斷的干擾，因此沒有據此宣稱完整冷建立比 FEM 快。

本頁的離線流程仍先建立完整原頻譜，再做壓縮。
另有[直接保存自適應插值與磁碟時間權重的實驗](native_spectrum.md)，
已通過含實際 BEM 插值區間的短程解／AD 驗證，降低首次建核的頻譜容量需求。
線上 FFT cache 仍須限制容量，且新 backend 尚未完成原 45 ms 驗證；
原 2 us／45 ms 四-worker 案例繼續使用既有 backend。
