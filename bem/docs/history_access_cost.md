# 原歷史卷積的資料重用成本

`tools/history_access_cost.py` 計算原 `BatchedOnlineConvolution` 的來源權重切片量與 kernel FFT 呼叫數。
它使用原 `cache_layout`，包含各 family 平分的 FFT 快取、有限容量的短 lag RAM prefix、
checkpoint 後空的 FFT cache，以及原程式仍會計算的未使用未來 target。
原求解器與執行中的完整案例都沒有修改。

## 已核對的範圍

[2026-11-04 控制資料](history_access_cost_results.json)：計畫時域為 4096 步，使用兩個 family 與兩個 source channel，
實際 append local indices 0–1696；在 386 保留原 pending／inputs 並清空 FFT cache，模擬 checkpoint 接續。
比較的區間為 386–1696，沒有將這項控制宣稱為完整計畫時域的驗證。
六種設定的切片 bytes、FFT 呼叫與 retained levels 均與原實際卷積器逐項吻合；
可觀測計算值逐位相同。獨立公共 helper 的五項測試也通過。
這是資料重用控制，沒有新的細網格物理解或 AD 收斂結論。

## 原細網格剩餘區間的估算

下表僅適用已驗證的 `(64,128)` 前綴：global next step 14451，第一個 correction 在 14065。
依原 full plan 22500 與原 append 行為，local indices 386–8435 共 8050 次。
所有列保留相同係數與歷史；16 GiB 陣列預算的列均通過原資源公式，
亦預留原 checkpoint 容量。這不是 process RSS 上限，未包含映射檔案頁與 OS cache。

| Prefix lags | FFT 上限 MiB（合計） | 每批 rows | 來源切片 GiB | kernel FFT 呼叫 |
|---:|---:|---:|---:|---:|
| 32 | 8 | 4 | 339.89 | 66766 |
| 512 | 8 | 4 | 206.51 | 66766 |
| 32 | 2048 | 4 | 240.66 | 8252 |
| 512 | 2048 | 1 | 206.51 | 32928 |
| 1024 | 8 | 1 | 172.10 | 266562 |
| 256 | 2048 | 2 | 238.78 | 16498 |

例如維持 32-lag prefix 與四 row，只將 FFT 快取由 8 提高為 2048 MiB，
可保存 32／64／128 block 的 transforms；來源切片由約 365 GB 降為 258 GB。
1024-lag prefix 的來源切片更少，但使用較多 RAM，且仍重算 kernel FFT。
一個 FFT 呼叫的 tensor 大小隨 block／rows 改變，不能直接用呼叫數比例推算時間比。

**實作決策**：原完整基準完成後，先比較同 row／同 prefix 的較大 FFT 快取，
再比較較大的 RAM prefix 與較小 row batches；以完整場、原始因果方程與導數門檻檢查每個版本。
實際排序仍需隔離量測，不能只按表格選定最快設定。

此模型不含全權重 SHA 讀取、初始化、prefix 建置、編譯或 source FFT 成本。
RAM prefix 的額外初始化切片量另存 JSON。實際磁碟讀取還受 file layout、page cache、
readahead 與記憶體壓力影響；以上並非磁碟流量或完整 runtime benchmark。
完整事件／FEM 一致性、時間／空間收斂與同精度速度／穩定性優勢仍待完成。
