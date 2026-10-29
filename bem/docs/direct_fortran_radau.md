# 直接反變換為 physical Fortran Radau 權重

2026-10-29 新增 [`direct_fortran_radau.py`](../tools/direct_fortran_radau.py)，
將已完成的 work-coordinate compact `F0/F1` 頻譜直接反變換為
physical、Fortran-order 的兩組時間權重。
原始建核器先寫 C-order 時間權重、轉成 physical，再產生另一份 Fortran-order 檔。
新元件省去最後轉換所需的一份完整時間權重暫存檔，保留全部 requested lags、
有限 DFT wrap 與原本的浮點運算次序。

這是獨立元件，**尚未接入正式建核器或目前的排程**。
正在執行的 FEM、等待中的 2 µs Radau 工作，以及已提交的 256 個 raw 頻點都保持原狀。
原 `/Volumes/Gauss-T7/tatva` 維持只讀。

## 數值與中斷接續

輸入必須是兩份已完成、C-order complex128 的 **compact work-coordinate** 頻譜，
並提供 shape、檔案長度、SHA256 與完成標記；不能直接傳 raw Radau eigenmode branches。
輸入在執行前後均以串流 SHA256 核對。

每次複製有限 row stripe 的半頻譜，使用原 `scipy.fft.irfft`，
保留全部 lags 的 radius amplification。時間權重的 physical normalization 是
`sqrt(w_i)*sqrt(w_j)`；wrap 則是原本的 `sqrt(w_i*w_j)`。
兩者的數學值相等，浮點位元可能不同，因此不能在整理程式時合併。
輸出直接寫入 Fortran-order NPY；每次 writable mapping 只有一個 column 的 row stripe，
不展開完整二階段時間權重，也不新增一份 C-order 時間檔。

每個 stripe 在 fsync 並核對實際寫入位元後，才原子提交 journal。
F1 stripe 與相應 wrap rows 一起提交。
中斷後重新計算未提交的 stripe；接續前核對所有已提交的 weight／wrap hashes。
數值輸入、時間計畫、line weights、NumPy／SciPy 或此元件原始碼改變時拒絕接續。
exclusive lease 防止兩個 writer；中斷於分配階段時只重建本 journal 尚未提交的輸出。
不知道結束時間的舊 attempt 仍標示為未完整計時，不推算或補造成本。

工作陣列預估先於 square normalization 配置，亦檢查全部輸出所需空間。
這是 array workspace guard，**不是 process RSS 上限**，也不包含呼叫端已持有的陣列。
輸入／輸出 hash 與驗證有額外 I/O 成本；目前沒有量測大型總時間改善。

## 已完成的控制

[完整紀錄與指紋](direct_fortran_radau_results.json)。

- 16 項測試通過，涵蓋兩種大小的全部 weight 檔案位元一致、wrap 位元一致、
  未提交 columns／已提交 stripes 後接續、分配中斷、已提交資料損毀、
  輸入／runtime 改變、bounded mappings、競爭 writer 與 preflight。
- 使用**未修改的原始 BEM 建核器**產生小型完整核心：Q12、h/contact/end/corner
  160 mm、11 個 retained unknowns、dt 2 µs、16 步、FFT 64、durable build。
  新流程兩組完整 physical Fortran 權重的檔案 bytes 與 SHA256 均相同。
- 同一核心的全部 wrap coefficients 與完整 causal actuator startup 均逐位元相同；
  lag-zero 原逆算子 composition 的 infinity norm error 為 `6.68e-11`。
- 新反變換本身含輸入／輸出核對約 `0.570 s`；整份 checker 約 `0.613 s`。
  這個案例只驗證儲存／算術一致性，不能用來宣稱完整事件加速。

本機測試與真實 BEM 控制依序使用一核心；執行時暫停同一個 FEM 及其 CPU quota，
結束後恢復原 PID，沒有同時啟動第二份大型模擬。
兩次等待共 `11.372667 s` 已包含在長程 FEM 的 wall interval，不能再相加一次。

## 完整 45 ms 的儲存影響

以下固定 531 unknowns、目前的 FFT padding，僅計 **array data 的邏輯 bytes**。
不含 NPY headers、startup、journal、filesystem allocation／reserve、FEM、原方程稽核或其他 run。
不能直接當作磁碟 free-space preflight。

| 時間步 | 舊 durable 峰值：raw + compact + 兩組 weights + 一組轉換暫存 | 直接輸出：仍保留 raw + compact + 兩組 weights | 省下轉換暫存 |
| --- | ---: | ---: | ---: |
| 2 µs，22500 步，FFT 65536 | 692.530 GiB | 645.261 GiB | 47.269 GiB |
| 1 µs，45000 步，FFT 131072 | 1385.038 GiB | 1290.501 GiB | 94.537 GiB |

因此**此元件單獨還不足以讓 1 µs 計畫容納於目前磁碟**。
若之後另實作可中斷接續的 raw → compact 原地轉換，兩份頻譜角色共用儲存，
再搭配本元件，以上主要 data 才會降至約 369.900／739.787 GiB。
同日後續完成了[可接續原地轉換元件](inplace_radau_compaction.md)與小型完整 BEM 核對，
producer 交接與正式整合仍未實作；不能刪除現有 prefix、忽略重建成本，
或將未執行的完整方案列為實測結果。
同日更後續已完成[整合 producer 與短程接續驗證](storage_radau_pipeline.md)，
完整 1 µs 事件尚未執行。

## 使用與後續

對已完成、work-coordinate spectra 仍存在的 Fortran Radau 核心核對：

```sh
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -B bem/tools/check_direct_fortran_radau.py PATH_TO_COMPLETE_KERNEL \
  --output bem/outputs/new-direct-fortran-check --max-workspace-mb 512
```

checker 要求全新 output，額外產生兩份完整 weights，不能在現有大型核心上未做容量評估就啟動。
它檢查原 case／source／seed／runtime，從幾何重建 line weights，不重新組裝頻率矩陣。
會核對全部 weight bytes、wrap、完整 causal startup 與 lag-zero composition；不執行非線性事件。

程式 API 為 `invert_physical_fortran(family_records, plan, line_weights, folder, ...)`；
其 `resume=True` 使用同一個既有 journal。checker 本身只接受 fresh output。

durable 原地轉換與獨立 producer 的整合見[後續紀錄](storage_radau_pipeline.md)。
仍需完成 macro time／contact／space 收斂、
完整 FEM 場量／前緣／波與能量核對、初始化 AD 的收斂，以及同精度穩定性或總耗時優勢。
權重位元相同並不代表上述目標已完成。
