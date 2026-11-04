# 全程失敗點的局部根與完整歷史恢復

2026-11-04：原版與短 lag cache 版都在 **第 14451 步／28.902 ms** 失敗，
residual 同為 **7.435954410063236 MPa**。兩份 capture 的全部 15 個 NPY members
逐 byte 相同，包含已接受的位移與接觸前綴。因此這次失敗並非只出現在 cache 版。
兩個排程均已結束；完整 45 ms 尚未重現，FEM／速度／穩定性目標仍未達成。
[結果、來源、配額與剩餘門檻](captured_radau_failure_results.json)保存本輪證據。

## 找到的是哪一個解

原方程、物理狀態、已接受歷史所形成的 forcing、float64 與 `(64,128)` 子步保持原樣。
原 1062 個 stage 未知數先作線性消去：完全弱化的外部點採 closed kinetic 候選分支，
未弱化的外部點採 stick 候選分支；保留點仍使用原完整接觸律，包括法向、張開與 stage 耦合。
**候選分支不能決定接受結果**；回代後必須通過原完整方程的 `1e-7 MPa` 門檻。
測試中刻意製造「縮減方程已收斂、外部候選分支錯誤」的情況，確認會拒絕。

只保留前緣兩個點的控制仍未找到根；人工 forcing 的有界曲線追蹤也未達目標。
後續保留全部 **13 個未弱化點、52 個法向／剪向 force corrections**，
固定 seed `20261104` 的 multistart Powell hybrid 在**第 8 個起始猜測**找到根：

- 原完整 JAX residual：**1.5395e-11 MPa**。
- 獨立 NumPy 重播兩段原 return map：**1.9721e-11 MPa**。
- 所有 stage 欄位有限；接觸皆受壓，沒有反向塑性滑移。
- 最大 shear 位移增量 0.028442 mm；最大累積滑移增量 0.031116 mm。
- 兩段總微步摩擦功 20.5883 MPa·mm²，各點的微步功總和皆非負。

這個根來自 **multistart hybrid**，並非人工 forcing 曲線成功到達目標。
位移與狀態增量需繼續做時間與子步控制；有限值、正摩擦功及小殘差
都不構成完整軌跡的能量平衡、唯一性或穩定性證明。
目前尚未提交該狀態或恢復完整 CQ 推進。

後續[初始化與完整前綴驗證](streamed_initialization.md)已完成：14451 筆原平衡均通過，
全部未來歷史與原逐步重播逐位相同，失敗步驟 forcing 也逐位相同。
已保存恢復所需資料；接回時間推進及不中斷軌跡控制仍待完成。

## 可重播的小型輸入

公開 [equation.npz](diagnostics/radau_step14451/equation.npz)約 8.49 MB，
僅包含失敗時的完整矩陣、forcing、前一步狀態與原參數；535 MiB 的已接受前綴仍保留本機。
同目錄的 `failure.json` 保存小型檔案 SHA 與原 capture SHA，
`candidate_roots.npz` 保存這次找到的完整 stage 位移。

```bash
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 JAX_ENABLE_X64=true \
python -B bem/tools/diagnose_radau_branch_continuation.py \
  bem/docs/diagnostics/radau_step14451/equation.npz \
  --output bem/outputs/replay-step14451 \
  --retain-all-unweakened --include-normal --random-starts 32
```

工具會檢查 SHA、重播原失敗 residual、核對縮減切線與獨立 `jacfwd`，
並保留每個起始猜測的完整方程接受結果。它不需要重建 bulk kernel。
這個案例的縮減切線相對差為 `2.63e-14`，線性消去 backward error 為 `4.50e-18`。
該數字是局部代數檢查，並非完整 BEM 誤差估計器。

## 完整歷史仍須恢復

`tools/restore_compact_history.py` 用原 append 演算法重播 dyadic frontier，
按原時間順序恢復未來 pending contributions、完整已接受 inputs 與 delayed F1。
12 項控制在多個直接／FFT 邊界、兩種 cache、transpose 及 zero-source rebase 下，
後續每步結果都與不中斷執行逐位相同。已消耗的 pending rows 不重建；
新的 FFT 統計也不能抹去先前已付費的成本。

這仍不是完整 PDE checkpoint。接觸狀態、載入停止規則、原始初始化 residual history、
來源 corrections 與保存的場必須共同恢復，並重現失敗時 forcing 後才能續跑。
原 capture 沒有保存每一步的迭代次數與 residual，不能把缺失值填成零而宣稱已稽核。

合併的 24 項局部控制通過；擴大候選分支後的 6 項相關控制亦通過。
先前一次診斷的 quota controller 身分檢查失敗，該次資源控制結果已保留為拒絕；
後續控制改由保留 `Popen` 子程序所有權的 parent 執行，平均 CPU 配額 3.8，
完整 package 指紋仍與原 kernel 相同。這些短程結果不是隔離的全事件速度比較。
