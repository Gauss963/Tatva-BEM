# 可選 JAX 建構器與獨立 checkpoint 複製

新增 `tools/prepare_jax_radau_resolvent.py`，讓完整 Radau 核建構使用已驗證的
JAX batch 組裝。原 producer、原 Tatva 與目前執行中的細網格快取都沒有替換。
完整 FEM 一致性與同精度的穩定性／總時間優勢仍未達成。

新工具使用私有 namespace，保留原 raw sampler、積分、接受門檻、checkpoint writer、
頻譜組合、所有因果 lag、shift wrap、啟動與輸出。AST adapter 只接入 backend、
來源／runtime 檢查、資源估計與結果標記，沒有修改原 module globals。
例外與 signal cleanup 共用同一組私有 journal／lease，避免留下錯誤的建構狀態。

每次建構另凍結 `jax-backend-context.json`、JAX／jaxlib 版本、x64、CPU devices、
thread environment 與所有 backend 來源。續跑不能默默改用不同 backend。
材料與幾何 AD 的公開介面仍未完成；此工具是固定 bulk 的建構流程。

另有[可選 general-LU／lazy Trace-Schur 建核器](trace_solve_policy.md)，
保留原 startup，已通過完整粗網格 kernel／滑動／初始化 AD。
細網格兩對頻率的求解時間改善尚未構成完整事件或 FEM 的速度證明。

## 獨立參考稽核

原 producer 的 final audit 會把 `transfer.prepared` 指給另一個直接 compliance solver。
如果只替換 prepared objects，該 audit 也會使用近似核。
新 `OriginalAuditTransfer` 會將 JAX wrapper 解回原 `PreparedGalerkinOperators`，
因此這條獨立求解路徑繼續使用 SciPy 核函數。
測試另外故意讓 JAX wrapper 的 `assemble` 無法使用，確認 audit 仍得到原始解。
composition tolerance 維持 `1e-6`，沒有放寬。

2026-11-01 後續 Linux CI 發現 lag-zero impedance 的四個小元素未通過原精度門檻。
現將直接 lag-zero 的兩個頻率保留在原 SciPy layers／Trace-Schur 路徑，raw 頻譜仍使用 JAX；
另要求其結果與原版逐位元相同。修正後 14 項測試通過，
[CI 失敗與修正證據](fem_modal_resolution.md)獨立保存；以下十三項為修正前本機紀錄。

## 續跑與複製的條件

同一 JAX 建構中斷後，仍由原 checkpoint writer 逐位元檢查已提交 prefix。
原版轉 JAX 則使用獨立工具 `tools/clone_jax_radau_checkpoint.py`：

1. 來源必須是未完成的原版 raw build，且其 constructor lease 已釋放。
2. 目標必須是新生成的 JAX `--durable-build --preflight-only` 資料夾。
3. 兩邊的物理問題、CQ plan、取樣設定、原始來源、NumPy／SciPy 必須一致。
   worker 與資源上限可另記錄；dt、積分及接受門檻不可改動。
4. 每個未完成 branch 的 committed prefix 必須等於保存的 sampler start。
   不在邊界就拒絕，不能繞過原逐位元重播檢查。
5. 來源檔以唯讀方式開啟，僅複製已提交區段。所有 chunk 的 SHA-256 在來源與目標
   都要驗證；NPY header、shape、dtype、長度保持一致，未提交 suffix 不繼承。

複製後保存 parent／template snapshots、原 attempts 與 receipts，並在新 preflight、
context、journal 記錄 handoff。來源快取不變；未提交工作量也不會被算成已保存數值。
新建構會繼續檢查這些不可變的 handoff 來源，沒有以「同一個算法」掩蓋 backend 轉換。
既有 checkpoint 的已知 producer hash migration 規則沿用，其他數值來源仍須完全一致。

流程是先建立新 preflight，再複製、驗證，最後用相同數值參數加 `--resume`：

```sh
PYTHONPATH=bem/src JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 \
  python -B bem/tools/clone_jax_radau_checkpoint.py \
  /path/to/stopped-original-prefix /path/to/new-jax-preflight
```

上列不是對目前執行中快取的操作指令。2026-11-01 實測時，原第一個 branch 已完成
32769 點，第二個 branch 提交 7744 點、保存 start 為 0，且 lease 仍由原程序持有。
兩項條件都拒絕接手。本次只做唯讀探測，沒有複製或遷移這個實際快取。
第二個 branch 已僅用 24 次 raw evaluation 進入大量插值／凝縮；此時更換徑向核
backend 可節省的剩餘成本有限，因此讓原程序繼續完成完整事件所需的核。

## 驗證與資源

第一輪八項測試通過（24.63 s），最終來源的十三項測試通過（73.86 s）：

- C／Fortran 完整 16-step kernel、所有 lag 與啟動對照獨立 dense original BEM。
- 在實際 JAX 建構中注入中斷，續跑的所有 NPY 與啟動陣列逐位元一致，verify-only 不改寫檔案。
- 實際 128-step 原版 prefix 複製、混合續跑後，所有 compact families、權重、wrap、
  lag-zero impedance、啟動陣列對照完整原版，維持 `rtol=1e-8, atol=1e-12`。
- 原版來源所有檔案保持不變；活躍 lease、改變 CQ plan、非邊界 prefix、損壞的
  committed bytes 及改動 backend context 均被拒絕。

這些是建構與儲存流程驗證。後續[混合快取的完整短程滑動／初始化 AD](mixed_radau_ad.md)
已另外通過，直接使用保存權重；細網格完整事件仍須完成，不能用此粗網格控制取代。

資源估計在原 array peak 上另加 `max(512 MiB, min(2 GiB, geometry/2))`，
作為 JAX runtime／compiler allowance，這不是 hard RSS cap。
同一 531 DOF、Q12、45 ms／2 µs 設定的估計 peak 為 **20099.50 MiB**，
包含額外 2048 MiB；原 18432 MiB 上限正確拒絕，20480 MiB 的 T7 preflight 通過。
預留建構磁碟需求 **693.2435 GiB**。沒有建立任何頻譜／權重陣列，完整 RSS 尚未量測。

第一次探測誤把目標放在內部磁碟，通過記憶體檢查後被容量檢查拒絕。
該失敗與耗時保留；改用 T7 新的 BEM 資料夾後才完成正確預檢。
受控測量沿用平均 3.85 CPU quota，原首 256 個已保存矩陣在成功控制前後 SHA-256 一致，
每次都恢復同一組原程序。

[結果與來源 JSON](jax_radau_producer_results.json)包含每次測試、失敗探測、來源快照及
controller。原 `/Volumes/Gauss-T7/tatva` 持續只讀。
