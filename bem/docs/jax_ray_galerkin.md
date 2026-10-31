# JAX 編譯 Galerkin batch 的控制實驗

此為可選實驗 backend。原長程 producer、已付款的原始核矩陣與原 Tatva 均未替換。
它的目的，是減少同一組細網格、Q12 積分方程的組裝成本，讓後續時間解析度研究更可行。
完整事件與 FEM 的一致性及同精度總耗時優勢，仍須另行驗證。

`tools/jax_ray_galerkin.py` 從現有 `PreparedGalerkinOperators.assemble` 擷取每個
batch 的方程，保留 singularity subtraction、解析 log corrections、反向 double layer
與原始 host scatter。核心函數在 module 範圍建立一次；頻率、材料數值與 table
coefficients 作為 dynamic JIT inputs。不同 batch/table shapes 的首次編譯另有紀錄，
不將每個頻率設為 static argument 或在迴圈重建 JIT function。

Bessel table 沿用既有 degree-24、off-grid 與 coefficient-tail 檢查。
任何 bin 被拒絕，整個頻率退回原始 factored NumPy 實作；這些經驗檢查不是一致誤差證明。
JIT 路徑以同一組係數同時計算 K0／K1。原 radial algebra 與九項小參數導數級數
直接沿用；為避免未啟用分支發生大冪次 overflow，僅對該分支的輔助級數輸入設安全值。
所有實際距離與啟用級數的輸入不變。

矩陣仍輸出 NumPy complex128；每個 batch 讀回 host 會等待 JAX 完成。
因此組裝計時包含首次編譯、資料傳輸及同步。wrapper 另記錄 JAX 啟動成本與 process peak RSS。
本介面沒有提供材料／幾何導數：將 table coefficients 視為固定值直接微分，
不會自動得到物理核函數對材料的完整導數。既有接觸根與初始化的 implicit AD 另行保留。

測試對照原始 SciPy 核，而非只對照先前的 table backend：

- 兩個完整物體的 V、H、D：`||A-B|| <= 1e-14 + 1e-11 ||B||`。
- 完整凝縮阻抗與所有邊界還原：Frobenius／infinity 相對誤差均不超過 `1e-8`。
- 小型完整 Radau contour、所有因果 lag、forward history 與 future transpose。
- 插值拒絕時的整頻率 exact fallback，以及原始 module globals 未變。

首次單元測試因測試包裝器漏傳 fixture，為 6 pass／1 fail；修正後為 7 pass。
失敗版本、日誌與耗時保留，不能算成通過的數值控制。

## 實際細網格結果（2026-10-31）

原 531 reduced DOF、outer/contact/end/corner = 10/2/0.015625/0.015625 mm、Q12，
45 ms、2 µs 的 M=65536 contour，兩物體完整組裝。每個模式交替順序重複兩次，
下列為中位數；JAX 欄包含每次重建 table 及首次使用 shape 的編譯。

| 物理模式 | 原始 SciPy／NumPy（s） | JAX（s） | 本次組裝時間比 |
|---|---:|---:|---:|
| slow branch，index 0 | 53.6774 | 15.3937 | 3.49× |
| slow branch，index 655 | 98.7105 | 15.1823 | 6.50× |
| fast branch，index 0 | 51.8456 | 13.4645 | 3.85× |

六次對照全部通過原 raw／Schur 門檻。最大 raw normalized error 為 0.00328991
（允許值 1）；阻抗／完整邊界還原的最大相對誤差為 2.81224e-13（允許值 1e-8）。
所有 table 均為 30/30 bins 接受；共 trace 九個 batch signatures，後續兩個頻率
未增加 trace。process peak RSS 為 3,925,360,640 bytes（3.66 GiB）。

原 geometry 準備 23.3222 s，新增 geometry 掃描 0.4141 s，wrapper 的 JAX／module
啟動 0.5385 s。完整數值 benchmark 528.9488 s；外層含單元測試、啟動與保護檢查
545.3909 s。這些開發暫停成本全部留在原 BEM 的 wall time 中。

控制期間原 producer、quota 與 owner 均暫停；實驗另用同一個平均 3.85 CPU
quota，BLAS=1，JAX=x64。原組裝是序列單頻工作；這不是原 producer 雙 worker
吞吐量的比較。只有兩次重複與三個頻率，也沒有完成事件計時或統計速度保證。
不可將表中的時間比解讀為完整 BEM 比 FEM 快。

可重跑相同數值實驗（另選全新 output，並確保其他重運算已暫停）：

```sh
JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=bem/src python -B bem/tools/benchmark_jax_ray.py \
  bem/src/finiteblock_bem/resources/cases/pmma_event16.toml \
  --output bem/outputs/new-jax-ray-study --indices 0 655 --branches 0 \
  --include-fast-zero --repeats 2
```

CPU quota 及完整環境設定保留於結果 JSON 的 controller source／release receipt；
上列只列數值命令，不代表單靠環境變數就已強制四核心上限。

## 完整短程滑動與初始化 AD

再使用原始 healed preload、完整 514 個物理 contour modes 加兩個獨立 lag-zero
modes，執行 128 步粗網格診斷。微步為 8/16；診斷載入速率 7812.5 mm/s，
明確不同於原案例的 20 mm/s。兩版本均產生 1.03382197996 mm 的最大累積滑移，
已進入非平凡弱化，而非以全黏著狀態通過控制。

全部 raw／Schur／因果權重門檻通過；全部階段位移陣列的相對差為 9.72486e-14；
含接觸狀態及微步摩擦功的整個 response 陣列相對差為 2.13851e-13，
這個混合陣列範數不是每個物理欄位的獨立相對誤差。五個參數梯度的相對差為
2.70060e-12，所有階段 actuator gradient 為 5.14417e-13。
七個 initialized JVP directions（3+3+1）與 VJP 的原混合門檻
`rtol=1e-7, atol=1e-18` 全部通過；沒有放寬門檻。

數值控制 55.7075 s，外層含啟動與保護檢查 58.5823 s。
這是 backend 接合與所選離散分支的梯度對照；沒有新增獨立完整軌跡 finite-difference
驗證，也未證明完整細網格事件、材料／幾何或 endogenous stopping-time 導數。
它不是與先前不同時間執行的 paired backend 的公平速度比較。

完整數值資料、各次成本、失敗版本、controller 與來源快照保留在
[結果 JSON](jax_ray_galerkin_results.json)。原 paid cache prefix 的 SHA-256 在控制前後
一致，完成後恢復同一組 producer／quota／owner PIDs，沒有將新 backend 注入舊程序。

本輪列出的外層成本合計 **631.9820 s**：首次失敗測試 11.3915 s、修正後測試
14.3923 s、含再次測試的細網格控制 545.3909 s、滑動／AD 控制 58.5823 s、
發布與來源／格式檢查 2.2250 s。它們是開發控制成本，不是完整事件的 isolated runtime。

```sh
JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=bem/src python -B bem/tools/check_jax_ray_nonlinear.py \
  --output bem/outputs/new-jax-ray-sliding-control
```
