# 可接續的 raw → compact 原地轉換

2026-10-29 新增 [`inplace_radau_compaction.py`](../tools/inplace_radau_compaction.py)。
它將獨立 staging folder 中已完成的兩個 raw Radau mode branches，
原地轉成 work-coordinate compact `F0/F1`，重用原來兩個檔案的 inode。
搭配[直接輸出 physical Fortran 權重](direct_fortran_radau.md)，
可避免另外保留兩份完整 compact 頻譜與一份時間權重轉換暫存。
全部頻點、時間權重與有限 DFT wrap 仍保留。

**目前完成的是兩個儲存元件與小型完整 BEM 核對。**
raw producer 的所有權交接與整合版執行器尚未實作。
現有 2 µs 排程、256 個已保存頻點與完整 FEM 均未改動或重啟；
原 `/Volumes/Gauss-T7/tatva` 維持只讀。

## 所有權與資料角色

初次呼叫 `compact_inplace(raw_records, plan, folder, ...)`，folder 只能包含
`work-family0.npy`、`work-family1.npy` 與可選的 lock。
工具拒絕含原 producer `preflight.json`／`sampling-journal.json` 的目錄，
亦拒絕 symlink、hardlink、不完整或缺少 raw branch role 的 receipts。
輸入必須在呼叫前完成獨立原算子稽核；本元件不重新組裝 BEM 或放寬原稽核。

初始串流讀取同時產生整檔與各 batch 的 SHA256，將 chunk manifest 綁定原始收據。
接續時固定原始 raw receipts、inode、計畫、batch、NumPy／SciPy 與數值原始碼。
已提交區域核對 compact hashes，尚未消耗區域核對 raw hashes；
只有已保存 write-ahead backup 的未提交區域可以包含中斷造成的混合資料。

**原檔名保留，但內容角色在完成後變成 compact `F0/F1`。**
之後必須使用回傳的 compact receipts；舊 raw sampler 不可再接續這兩個路徑。
原 raw SHA256 留在 journal 作 provenance，不能拿來當目前內容的 SHA256。
正式 producer 之後需要可驗證、可接續的交接程序，不能直接在現有 producer 目錄呼叫本元件。

## 每批交易與資源

每批先保存兩個原始 raw arrays 至 bounded、未壓縮 NPZ，fsync 後原子發布 pending receipt。
再使用原 `compact_coefficients` 及相同 endpoint symmetry 規則，依序覆寫兩個 families。
每份資料 fsync 並核對實際位元後，才一起提交該批；最後刪除過期備份。
檔案 rename／journal publication 亦 fsync 所在目錄。

中斷於兩次覆寫之間時，接續從 raw backup 重算整批。
即使 pending 區域的 bytes 已部分損毀，也不把它誤認為完整 raw 或 compact。
其餘已提交／未消耗資料或 backup 損毀時，會在任何修復與 journal 更新前拒絕。
最後的串流整檔收據同時重新核對每批已提交係數，避免只替損毀檔案產生新的 SHA256。

預設每批 8 matrices。preflight 亦按 batch 數預留 journal／JSON metadata 空間，
在建立長時間計畫的大型 Python chunk manifest 前檢查預算。
531 unknowns 時，workspace 預估與 bounded backup
見[資源紀錄](inplace_radau_compaction_results.json)；這不是 process RSS 上限，
亦不含呼叫端持有的模型。備份、chunk hashes、journal 與 fsync 都有成本，
尚未量測大型總時間改善。
不明結束時間的舊 attempt 保留為未完整計時；外層 invocation interval 仍需另外保存。

## 已完成驗證

[完整證據](inplace_radau_compaction_results.json)。最終版本有 20 項新測試，
加上既有 direct-weight 的 16 項測試，共 **36 項通過**。

- 三組不同 batch／大小的全部 compact coefficients，與原 combination 算術逐位元相同。
- 三個子程序直接 `os._exit(73)`，跳過 Python 清理，分別停在 pending publication、
  第一個 family 覆寫後，以及完整 chunk 提交後；接續結果均逐位元相同。
- 另在 pending 區域刻意寫入錯誤複數資料，仍從 backup 正確重算。
  已提交 compact、尚未消耗 raw suffix 或 backup 本身損毀則拒絕。
- 核對 source／runtime／batch 改變、hardlink／producer 目錄拒絕、workspace preflight、
  bounded writable maps，以及完成前的既有 chunk 損毀偵測。

使用原始未修改 BEM producer 已建立的 Q12、11 unknowns、dt 2 µs、16 步、FFT 64 核心，
複製它的小型 raw branches 至全新 staging 執行完整核對。
全部 compact 檔案 bytes 與原來相同，兩個 inode 均保留。
接上直接 IFFT 後，兩組 physical Fortran weight 檔、wrap 與完整 causal startup 均逐位元相同；
lag-zero inverse composition error 為 `6.68e-11`。
原 kernel 在前後所有收據核對中保持不變。
這些控制沒有執行完整非線性 event，也沒有證明 FEM 精度、梯度收斂或數值穩定性優勢。

測試與實際 BEM 控制一核心依序執行，期間暫停並恢復同一個長程 FEM 與 CPU quota。
第一版控制與最終版重核對的成本均保留，等待時間已包含在 FEM wall interval，不能再相加。

## 儲存與尚未接入的部分

固定 531 unknowns、45 ms，以下只計兩份共用角色頻譜與兩份時間權重的 array data：

| 時間步 | 頻譜＋權重 data | 舊 durable data 峰值 | data 差額 |
| --- | ---: | ---: | ---: |
| 2 µs | 369.900 GiB | 692.530 GiB | 322.630 GiB |
| 1 µs | 739.787 GiB | 1385.038 GiB | 645.251 GiB |

以上是已驗證元件所需布局的**邏輯估算**，並非完整 producer 實測。
尚需另加 backup、headers、startup、journal／filesystem reserve 與完整原方程稽核空間。
現有 2 µs durable 建核完成後也會占用大量磁碟；若兩個完整核心同時保留，
不能僅依上表宣稱 1 µs 可以啟動。目前沒有刪除任何現有頻譜或變更排程。

小型完整控制可從 repository root 執行：

```sh
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -B bem/tools/check_inplace_radau_compaction.py PATH_TO_COMPLETE_DURABLE_KERNEL \
  --output bem/outputs/new-inplace-check --max-workspace-mb 512
```

checker 預設 validation disk limit 2 GiB，避免意外複製大型核心；
需要時可明確設定 `--max-validation-disk-mb`。
它複製小型 raw branches 供驗證，正式 `compact_inplace` 元件本身不複製完整頻譜。
接續 API 使用原 raw receipts、同一 folder 與 `resume=True`；checker 僅接受 fresh output。

下一步是完成 producer／staging 的交接及整合，保留原始 sampling、算子稽核與計時，
再做完整 macro time、contact、space 與 initialized AD 收斂。
完整 FEM 場量／前緣／反射波與能量，以及同精度穩定性或總耗時優勢仍未達成。
