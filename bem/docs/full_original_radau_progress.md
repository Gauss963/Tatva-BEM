# 原細網格全程 Radau：核已完成，非線性事件執行中

狀態快照：2026-11-03。原 case 為 `local-mus08-muk06-gap0-v2-ev16`，
完整時間 45 ms、dt=2 µs、22500 步、531 個 history DOF、265 個接觸節點。
原 `/Volumes/Gauss-T7/tatva` 保持只讀；此頁不代表已通過 FEM 一致性或速度／穩定性門檻。

## 已完成的核

原版 SciPy／Trace-Schur 建核於 **2026-11-02 04:25（台北）**完成，沒有改用後續可選 backend。
兩個 raw branches 與兩個 compact families 各完整提交 32769 個 FFT 頻率矩陣，
兩份 float64／Fortran 時間權重均保存全部 22501 個 lag，未截斷歷史。

- branch 0／1 分別有 3265／144 次 source evaluations，每個 FFT 網格點均轉換輸出。
- 12 個獨立 fresh modal checks 的最大相對 Frobenius 差 **5.16e-13**。
- 6 個完整 coupled-stage checks 的最大相對差 **4.11e-13**。
- lag-zero inverse composition infinity error **1.15e-11**，原門檻 `1e-6`。

這些是取樣與線性建核檢查；完整時間域的原始因果方程 audit 仍需等待非線性軌跡。
既有 Chebyshev 接受檢查也不是整段頻譜的嚴格誤差定理。

本次續建程序 wall time **263442.010 s（73.18 h）**，包含先前的暫停與序列控制；
更早的已付費 prefix 成本另有繼承紀錄。此數字不是隔離的純建核 benchmark，
目前不能宣稱本次原版 BEM 的總時間優於 FEM。
核內記錄的 inverse FFT 約 9302.94 s、權重轉換與完整 startup 約 21194.61 s，
因此不能只用抽樣 raw 組裝或線性求解的改善比推定全程加速。

## 啟動全程事件時的資源修正

原排程第一次模擬在任何時間推進前被拒絕：array estimate **13294.0 MiB**，
高於原設的 12288 MiB。沒有覆寫這個失敗紀錄，也沒有因此重建已完成的核。
新排程僅將 `simulate_compact_radau.py --max-memory-mb` 提高到 **14336 MiB**。
這是估算門檻，不是 hard RSS cap；主機為 24 GiB，執行仍受平均 3.85 CPU quota 控制。

所有數值來源與原保護輸入 hash 重新核對。
新 driver 直接保留原排程從 `runs=[]` 開始的 suffix，只有該 resource argument 改變；
物理條件、原載入、f09 seed、FEM-body active-set 初始化、FFT rows=4、cache=8 MiB、
root recovery、kinetic seed、完整 horizon 與所有接受門檻均未改動。

第一次新 driver 的 runtime guard 曾因 NumPy-only kernel metadata 沒有 `jax` 欄位而退出，
尚未建立數值工作。已保留該失敗，改從已驗證的 JAX context 核對 JAX／jaxlib 版本；
NumPy／SciPy 與 host platform 仍對照完成核。

## 正在進行與尚待驗證

22500 步事件已啟動，權重完整 SHA 校驗、原初始平衡與初始狀態保存均已完成。
執行中程序的 stack sampling 確認正在執行 JAX CPU kernels 與線性求解；
沒有逐步百分比輸出，不能以終端靜默判斷停止。

原排程會依序執行：

1. 全程 `(64,128)` 與 `(128,256)` 摩擦子步模擬。
2. 全時間域原始因果 RK 方程、完整場的子步 refinement 及 FEM 比較。
3. 帶符號 bulk port work；前述 gates 通過後，才進行兩組全程初始化 AD 與梯度 refinement。

獨立 macro-time／space／FEM 參考收斂及同精度總時間或穩定性優勢仍需完成。
原 FEM dt/8／dt/16 的未收斂證據不因建核完成而消失；不過濾目標、不以平滑曲線代替穩定性證明。
[完整核 metadata、hashes、失敗與 recovery source、執行快照](full_original_radau_progress_results.json)
記錄本頁所涵蓋的證據與範圍。
