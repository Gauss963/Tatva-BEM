# 完整初始化歷史與失敗前綴驗證

2026-11-04：已保存原始初始化 residual 的全時間域影響，並完成原細網格的
**14451 筆已接受狀態與未來 CQ 歷史驗證**。
失敗步驟的 forcing 重新計算後與原 capture **逐位相同**。
這些資料尚未接回完整時間推進；45 ms／FEM 與速度／穩定性門檻仍待完成。
[來源、數值、執行成本與配額](streamed_initialization_results.json)保存本輪結果。

## 初始化 residual 保持在方程中

原 body lift 的平衡誤差雖小，程式仍保留其完整因果歷史。
`tools/stream_initial_residual.py` 使用相同公式與全部 lag，改按兩份 Fortran 權重的
物理 column 順序投影，再作原累積和與 lag-zero 扣除。
SHA 同步涵蓋每個完整 physical column 與 NPY header，兩份原權重的完整 checksum 都通過。
沒有重建頻率核或重算靜態 body stiffness。

初始 lift 由原 `dynamic_refined_reference` builder 重新載入，核對原 case、完整 package、
f09 float64 seed、原壓力／剪力及靜態平衡門檻。
輸出涵蓋 **22501×2×531** 個 float64 值；投影與輸出 array payload 約 456 MiB，
file-backed 權重仍可能增加實際 RSS，這不是作業系統記憶體上限。

本次實際建構 wall time **712.929 s**、CPU time **121.769 s**，
包含完整約 100 GB 權重校驗、讀取、投影與保存；平均 CPU 配額為 3.8。
這是準備特定初始狀態的成本，不能拿它與先前整場 wall time 相除而宣稱全事件加速。
浮點累加 layout 有變，因此需另外驗證實際前綴與失敗方程。

四項小型控制核對原 row-tiled 公式及獨立展開的原始 causal stage kernel，
涵蓋 C／Fortran 磁碟來源、完整歷史與極短 horizon。

## 原細網格的實際檢查

`tools/validate_captured_radau_prefix.py` 對同一原始 535 MiB capture：

- 再次校驗完整原權重、lift、seed、初始 force 與完整 affine cache。
- 按原 endpoint loading-stop 規則重建所有 correction sources 與 actuator。
- 獨立重算全部已接受狀態的原平衡；最大 residual **6.95788e-8 MPa < 1e-7 MPa**。
- 以原 append 逐步重播，與 frontier 恢復的**全部未來 pending、完整 inputs 與 delayed F1 逐位核對**。
- 下一步 14451 的原 forcing **逐位相同**，以 reduced weights 正規化後的差為零。

原第一個 correction 在第 14065 步，因此 active prefix 有 386 筆；
checkpoint 保存局部 pending 的 386–511 rows，其後精確為零的 buffer 部分可在載入時配置。
完整 accepted inputs 與原核仍保留；這不是將實際非零反射歷史截斷。
本輪驗證 wall time **567.522 s**，亦計入成本。

`history_state.npz` 另保存 actuator、重新計算的 prefix residuals、初始化 forces 與 delayed state。
缺失的原 Newton iteration counts 明確使用 **-1**，不把它們填成已知零次。
另有三項控制核對 hold 從接受端點的下一個 interval 開始、diagnostic ramp 及變動初始狀態的拒絕。
最終七項相關控制通過；原 package 的全部 26 個來源與已完成核保持一致。

## AD 與後續推進

此 affine cache 與 checkpoint 是**固定初始狀態的 primal 資料**。
參數改變時須重新驗證 cache，初始化的 implicit sensitivity、初始 residual 的導數，
以及前綴每一步的狀態／歷史導數都必須保留。
將 checkpoint 陣列當作與參數無關的常數，會得到不完整的梯度。
同樣地，局部縮減方程中的 NumPy 消去矩陣不能在參數微分時視為常數；
恢復根後的參數導數應使用原完整 residual 的 implicit AD。

接回模擬器前，仍需完整讀回與不中斷執行的 coarse trajectory 控制。
之後才用已恢復的歷史與[原完整方程的候選根](captured_radau_failure.md)繼續細網格事件，
保留子步、原因果方程、FEM 完整場、初始化 AD 與時間／空間收斂門檻。
