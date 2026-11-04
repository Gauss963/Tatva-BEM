# 原細網格全程 Radau：核與首組 45 ms 已完成

**2026-11-04 最新結果：**第一組 `(64,128)` 已恢復並完成 22500 步／45 ms，
但六個 FEM 接觸場比較只有摩擦係數通過原 5% 門檻。
第二組 `(128,256)` 在第 14450 步失敗，其局部根已獨立驗證；
匹配的前綴驗證已通過，首組全程原始 RK 稽核仍在進行。
[完整結果與圖表](recovered_full_event.md)取代下方舊快照的執行狀態。
其餘內容保留為建核與先前失敗的歷史紀錄。

狀態快照：2026-11-03。原 case 為 `local-mus08-muk06-gap0-v2-ev16`，
完整時間 45 ms、dt=2 µs、22500 步、531 個 history DOF、265 個接觸節點。
原 `/Volumes/Gauss-T7/tatva` 保持只讀；此頁不代表已通過 FEM 一致性或速度／穩定性門檻。

**2026-11-04 更新：原基準也已在第 14451 步結束，與 cached 版有完全相同的
失敗殘差及 15 個 capture members。** 全部相關原程序均已終止，不能再當作執行中工作。
[局部求根與完整歷史恢復](captured_radau_failure.md)記錄新進展：保留 13 個未弱化點及
法向耦合的固定 seed multistart 已找到原完整方程的根，JAX／獨立 NumPy 殘差約
`1.54e-11`／`1.97e-11 MPa`。完整 CQ 尚未恢復，不能以此宣稱 45 ms 完成。
現已[驗證全部前綴與未來歷史](streamed_initialization.md)，失敗 forcing 逐位相同；
讀回與接入時間推進仍待完成。
下方 11 月 3 日的啟動、暫停與恢復敘述保留為歷史紀錄。

**11:25（台北）結果更新：cached 的第一組 `(64,128)` 完整案例在第 14451 步、
28.902 ms 失敗。** 平衡 residual 為 **7.435954410063236 MPa**，未達原 `1e-7 MPa`。
已接受前綴與失敗方程保留為 535 MiB 的 `failed_equilibrium.npz`；
目前這份檔案不是可直接續跑的完整 CQ checkpoint。沒有進入第二組細網格子步、
原始因果 audit、全程 FEM 對照或 fine AD，不能宣稱全事件完成。

失敗階段 wall time 3279.905 s，包含 prefix 建構 213.979 s；不是隔離效能比較。
排程的 `finally` 已在 **11:25:17** 依序恢復原 numerical／quota／owner 的同一組程序，
暫停共 3279.982 s。原基準繼續執行，不能預先假定它會得到相同失敗。
[terminal result 與恢復 receipt](history_prefix_full_event_progress_results.json)保留實際結果；
下方 10:30 的敘述是歷史啟動快照。

失敗方程的局部重播已精確重現上述 residual，正在使用相同物理狀態、forcing、
子步與原門檻診斷求根策略。這不會提交 trial state 或重新啟動 CQ。
另有[保留局部 AD 的切線組裝](identity_stage_tangent.md)已通過粗網格完整解／初始化梯度
及細網格代數控制；它尚未用於此次失敗的全程案例。

10:30（台北）更新：在[完整 coarse primal／初始化 AD 及實際 CLI 核對](history_prefix_cache.md)
與數值來源 `983c377` 完整 CI 通過後，啟動相同原始全程的可選短 lag cache。
prefix RAM 配額為 144 MiB／32 lags，總 array budget 14336 MiB 中，
原 planner 可用 14192 MiB；其餘物理條件、兩組摩擦子步、h／dt／Q12、
原始因果 audit 與全部 FEM／梯度門檻不變。
原程序 group 在同一記憶體狀態暫停，待新排程結束或失敗即恢復，
兩者的既有成本與暫停時間均保留。下面原排程敘述及 JSON 為較早快照。
[新的 cached 全程執行快照](history_prefix_full_event_progress_results.json)
另記錄目前程序、原排程暫停現場及完整 CI；兩次細網格初始 lift 的 array members 已逐 byte 核對一致。

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
