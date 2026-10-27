# 與原 FEM 接觸節點一致的完整事件控制

這輪目標是減少 bulk 空間離散與接觸演化時間的差異，仍使用原 event-16 的
有限雙 block、材料、friction parameters、restart 與停止載入政策。
目前只完成 grid check、preflight 與序列排程，不能當作完成的物理結果。
已完成的六組舊 bulk 控制與原始方程稽核見
[fem_initialized_completed_results.json](fem_initialized_completed_results.json)。

## 已核對的接觸網格

原 FEM h2 與 float64 seed 都有 241 個節點 `y=0,2,...,480 mm`。
新 BEM 使用 `h=10, contact_h=2, contact_end_h=2, corner_h=2 mm`，清除額外
restart refinement knots。其接觸 bulk trace 也有 241 個節點、483 retained DOFs。
逐點座標與 FEM／seed 最大差為 2.8422e-14 mm；五個座標有浮點運算順序的
末位差，因此不宣稱 bitwise coordinate identity。沒有調整原始座標或 seed。

`subdivisions=1` 使用相同節點的 nodal contact rule。先前增加 subdivisions
只增加接觸狀態積分點，保留原 137-node bulk trace，無法解析新增的 bulk 模式。
這一控制也不代表外邊界已收斂；外邊界 refinement 和 bulk dt 仍需另外檢驗。

## 執行順序與驗證條件

本機 observer PID 92049 等待現有完整 AD／layout study 完成；數值工作不重疊。
來源、輸入 metadata 及所有必要 helpers 在排程時保存 SHA-256，改動即停止。
新建核與 storage source `d7cedee` 已通過
[Python 3.11／3.12 CI](https://github.com/Gauss963/Tatva-BEM/actions/runs/40440738067)。
四核心 thread 設定是 `workers=4, BLAS=1` 與相同 JAX 設定，macOS 沒有硬 affinity。

1. 先以 241-node 接觸 trace 做 outer h20／h10／h5 靜態 lift，與 FEM 初態比較；
   再做 h2／h1 的三個 continuous-time 線性頻率響應，尚未引入 CQ。
   初態相對 FEM h2 全界面壓力／剪力差超過 5%，h10／h5 靜態差超過 1%，
   或三個 h2 線性響應的任一場／platen 相對差超過 5%，就先停止大型建核並檢查原因。
   這些是節省無效計算的工程篩選條件，不是論文誤差界，也沒有取代最終事件驗收。
2. 在原 fine2/Q12／2 us／45 ms grid 全新建核，原 spectrum／weights 不作為產生資料。
3. 每個頻率核對新舊 work resolvent 及 `A_old * L_new - I`；保留原有容差。
4. 完整重跑原 Γ1/coarse-P1 與 Γ8/original-P1，核對所有 saved fields，
   包含 interior histories、初始位移和 bulk force；不是只看 endpoint。
5. 前述等價性通過後，建立 241-node 接觸 bulk kernel，完整推進 N64 與 N160。
   兩者 bulk dt 同為 2 us；局部步長分別為 31.25 ns 和 12.5 ns。
   原 FEM dt 為 12.4929286 ns；局部子步接近它不代表 bulk 時間離散已等價。
6. 對兩條新軌跡重新做 original causal equation audit，再與 FEM h2／h1 的
   true float64 final_state、事件及保存歷程比較。任何 gate 失敗就保留結果並停止後續工作。

完整工作含所有 lags，沒有尾端截斷、材料調參、人工阻尼或端點排除。
使用 [raw-layer producer](raw_layer_sampling.md) 與經 bitwise read-back 的 Fortran NPY。
較連續的讀取能否減少總時間，要計入建核、轉換、hashing、初始化、JIT、推進及輸出。

## 資源與重跑

241-node kernel 的實際 preflight：峰值磁碟估計 206,391,372,400 bytes，
包括驗證完成前同時保留 C／Fortran 權重；完成後產物約 164.40 GB。
最大 explicit array 估計 12410.14 MiB，並非程序 RSS 上限；OS file pages 與庫快取另計。
原方程 audit 另存完整 operator spectrum。資料只放新 output 與 T7 sibling cache。

先以新路徑 preflight；正式建核移除 `--preflight-only`：

```sh
PYTHONPATH=bem/src FDBEM_CORES=4 OPENBLAS_NUM_THREADS=1 python -B \
  bem/tools/prepare_raw_layer_resolvent.py SOURCE_KERNEL \
  --dt 2e-6 --duration .045 --workers 4 --fft-rows 2 \
  --h 10 --contact-h 2 --contact-end-h 2 --corner-h 2 --clear-restart-knots \
  --weight-layout fortran --max-workspace-mb 6144 --max-memory-mb 16384 \
  --preflight-only --output FRESH_KERNEL_DIRECTORY
```

建核完成後，各自使用新 output 跑 N64／N160，例如：

```sh
FDBEM_CORES=4 bem/run.sh refined-contact-simulate COMPLETED_MATCHED_KERNEL \
  --restart-seed ORIGINAL_FLOAT64_SEED --subdivisions 1 \
  --seed-interpolation original-p1 --static-initialization fem-active-set \
  --constitutive-substeps 160 --kinetic-root-seed --output FRESH_PRIMAL_DIRECTORY
```

本機排程來源為 `bem/outputs/raw-layer-contact-study-20261027-v3.py`；狀態及每階段
command、exit code、計時、RSS samples 另存 `bem/outputs/raw-layer-contact-study-20261027-v3/`。
v1 尚在等待、沒有數值子程序時已由 v2 取代，transition record 保留在 v1 的
`superseded.json`。v2 增加建核前原 bulk 及新模型的時空場比較，並等待新工具的
`0d7f860` CI 已通過。v3 再將初態／空間／線性控制移到昂貴建核之前；
v2 同樣在沒有數值子程序時由 v3 取代，transition record 保留在 v2 的
`superseded.json`。既有 initialized AD 程序未停止。
這些大型本機產物不提交 Git；實作、測試與重跑設定保留在倉庫。
原 `/Volumes/Gauss-T7/tatva` 全程只讀。同精度 FEM 優勢仍未驗證。

## 起跑前就存在的空間差異

另已擷取 FEM h2／h1 最後一個 normal-phase static bulk frame，依 step ID
核對相同 high-rate frame。pressure 使用 strength／mu；shear 用同一 saved frame
的 paired displacements 和輸出 plastic offset 重建。healed memory 均為零。
這些是 float32 保存資料，不是假稱重新取得 true float64 初態。

|既有 BEM 初態|對照 FEM|壓力 P1 L2 差|剪力 P1 L2 差|
|---|---|---:|---:|
|Γ1、coarse-P1 seed|h2|19.52%|12.83%|
|Γ8、original-P1 seed|h2|32.96%|16.22%|
|Γ1、coarse-P1 seed|h1|9.09%|7.59%|
|Γ8、original-P1 seed|h1|23.04%|10.60%|

兩組 BEM 都是先前已完成的 healed-static controls，保留同一 137-node bulk trace。
此表不是單一變數的 refinement 實驗：seed 插值和接觸狀態積分也不同。
完整來源、frame／step、snapshot SHA-256 與儲存界見
[initial_contact_snapshot_results.json](initial_contact_snapshot_results.json)。
primary pressure 的 strength／mu rounding bound 最大約 0.000123／0.000171 MPa，
saved shear reconstruction bound 約 0.0182 MPa。額外的 displacement／coordinate
pressure reconstruction bound 較鬆，不能混稱 primary target 的不確定度。

以上差異發生在 CQ 動態推進之前。因此 bulk／初態空間差異不能只靠縮短 dt 消除；
是否解釋動態誤差的主要部分，仍須由新網格的完整事件檢查。沒有排除端點。

## 完整保存資料的時空誤差

`compare_refined_fem_history.py` 補上場與 scalar histories 的定量比較。
它在兩模型所有保存時間／空間節點的聯集上，逐塊精確積分 tensor-product P1
插值的平方差，不靠擬合平移、重設 rupture time 或外插。共同區間、保留的節點數
與最大插值場差都另行記錄；保存點之間或未保存的高頻振盪仍沒有誤差界。

FEM 時鐘由實際 `dt`、pressure steps、shear step IDs 重建，先核對每個
float32 timestamp 的 rounding。真實 h2 的 9000 筆剪切時間核對已通過，
保留 normal-phase 略微超過 2 ms 的差，沒有把第一個 shear sample 平移到零。
共同區間從約 17.93 ns 開始至 45 ms；這不是原始連續解的完整區間證書。

保存場包括 cumulative／plastic slip、mu、strength；pressure 使用保存的
strength／positive mu，儲存誤差另由 true float64 endpoint comparison 檢查。
工具不以 strength 代替局部實際 shear；高率 FEM 資料未保存該場，僅比較已存的
平均 shear history，局部 endpoint shear 仍用獨立 float64 final_state 核對。

本機五項控制已通過：獨立逐 cell Gauss 積分、解析 bilinear 場、內部脈衝、
零 reference／不交區間，以及實際格式的 CLI 讀寫與 clock rounding guard。
新工具的真實完整數據誤差尚待依序執行，不將單元測試當作 FEM 一致性結果。

```sh
PYTHONPATH=bem/src python -B bem/tools/compare_refined_fem_history.py \
  COMPLETED_BEM_RUN COMPLETED_FLOAT64_FEM_RUN --output FRESH_COMPARISON_DIRECTORY
```
