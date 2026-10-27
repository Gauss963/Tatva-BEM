# 用便宜的 Green 函數建議 raw 取樣的起始分段

目前冷建核最昂貴的部分是 raw Galerkin 矩陣組裝。
原 sampler 從整個半圓開始嘗試插值；一個區間失敗後才二分，
所以被拒絕區間的完整矩陣取樣也已付費。
新實驗先用相同材料的六個距離、三族 scalar Green primitives 找到便宜的 advisory partition。
這些值不會拿來代替 body matrices；只把區間索引交給真正 sampler 作為起點。

每個實際矩陣區間仍執行原 raw-family check、Chebyshev tail check、
獨立 condensed-output check。若不通過，照原程式細分或直接重算。
所有 required FFT indices 必須依序、恰好保存一次；缺洞、重疊、亂序都拒絕。
degree、raw/output tolerances、積分、自由度、完整歷史與摩擦律不改。

為保留正在長時間執行的 source，實驗 adapter 只在私有函式副本中替換
`stack = [(0, half + 1)]` 的值。移回原值後 AST 必須完全相同，
原檔案、module globals 與函式 API 不變。這沒有加入新的誤差定理；
scalar 建議對真正 body operator 的充分性仍由全部原檢查決定。

## 真實粗網格矩陣的完整半圓控制

h160／contact160／Q4 的兩有限體，共 11 個 retained DOFs，
使用目前 fine candidate 相同的 dt、65536 FFT size 與 radius。
兩條路徑均實際計算所需的 32769 個 [0, π] 頻率點；另一半由共軛對稱決定。

|項目|原全域二分|advisory 起始分段|
|---|---:|---:|
|真正 raw body 矩陣組裝呼叫|3982|2144|
|拒絕區間|82|5|
|接受區間|79|79|
|直接計算的 grid points|128|128|

實際矩陣呼叫減少 **46.16%**。
兩份保存的 `32769 × 11 × 11` complex128 頻譜逐位元相同。
另選十個頻率直接重新組裝；最大 normalized error 為 **1.24e-5**，
其中 1 才是原 output 容許上限。
[完整結果與來源](seeded_raw_matrix_screen_results.json)保存設定、原門檻、
每個檢查與頻譜 SHA；內含 fine resource preflight，仍是原 17.36 GiB 陣列估計。

這是粗網格的真實矩陣與整個 required contour 控制，不是 scalar-only 測試。
但它沒有建立 fine time weights、45 ms nonlinear trajectory 或 FEM 誤差證明。
量測與自有建核重疊，應比較呼叫數，不能用這份 diagnostic wall time 宣稱正式速度比。

## 使用方式與後續門檻

預設 trace producer 不啟用這個實驗；明確加上旗標：

```sh
PYTHONPATH=bem/src python -B bem/tools/prepare_trace_raw_resolvent.py SOURCE_KERNEL \
  --primitive-partition --output FRESH_KERNEL \
  --dt 2e-6 --duration .045 --h 10 --contact-h 2 \
  --contact-end-h .015625 --corner-h .015625 --clear-restart-knots \
  --quadrature 12 --degree 16 --workers 2 --weight-layout fortran \
  --max-workspace-mb 10240 --max-memory-mb 18432
```

`--partition-distances` 可明確指定 advisory 距離；preflight／完成檔會保存 helper SHA，
完成的 sampling record 另保存實際分段、proxy 成本與真正矩陣呼叫數。
加 `--preflight-only` 不會啟動完整建核。
回歸測試涵蓋錯誤的 advisory 仍須細分、全部 emissions、原 globals、
C／Fortran 全部時間權重及 causal startup；reference 為逐頻率重組的獨立 dense BEM。

切換候選長建核前須通過 CI、fine resource gate 與既有 static／linear gates。
任何被替換的部分建核和已付費時間都保留、累加，不把中止成本當成零。
新 kernel 完成後，原始因果方程稽核、完整 float64 FEM 比較、
子步／時間／空間收斂與 fine AD 仍是必要工作；本項不替代這些門檻。

切換時另保留已完成的細網格頻譜前段，作為新建核的早期 rejection gate。
`compare_raw_kernel_spectrum.py` 的 `compare_emitted_prefix` 要求 caller 提供已寫完的數量，
只讀對照已保存的區間；除 Frobenius 差，也計算 reference solve composition，
避免很軟方向上的誤差被全域範數掩蓋。
streaming SHA 僅涵蓋 NPY header 與這段資料，live candidate 會再次讀取相同前段核對 SHA。
它不把部分檔案標成完整 kernel；此反解仍來自 reference approximant，
不能替代獨立的原始算子稽核。回歸測試明確驗證：未寫完的 tail 不可被當成已驗證，
且全域範數很小的弱方向誤差仍會使早期檢查失敗。

## 2026-10-28 的實際候選切換

CI 與資源檢查通過後，已啟動 trace＋advisory 的 fresh fine kernel，
後接同一 45 ms N64／N160 事件、完整原方程稽核、真正 float64 graded FEM 比較與 gated AD。
舊 joint 建核未完成，但原檔案及前 1024 個已寫完的細網格矩陣保留，
以 header＋prefix SHA 固定，供新建核達到該區間時自動核對。

第一次 launcher 結束後，工具亦結束其子程序；實際 PID 缺失已確認，
現改成三個獨立、持續存在的 exec sessions，並以後續 `ps` 觀測核對建核 child 正在運算。
這次啟動失敗與恢復期間採保守 elapsed upper bound 記帳。
連同先前兩次未完成建核，帶入的新路徑已付費時間上界為 **35614.68 s（9.89 h）**；
沒有刪除中止成本、暫停時間或錯誤啟動紀錄。
[切換、成本與存活觀測](fast_route_transition_results.json)保存具體來源。
此時新的完整動態精度與總耗時優勢仍未驗證，不能把這份操作紀錄視為完成證明。
