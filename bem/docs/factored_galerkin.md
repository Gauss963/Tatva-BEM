# Galerkin 積分收縮的局部效能控制

`tools/factored_galerkin.py` 把四個 quadrature contractions 中不依賴積分點的
基底因子移到加總外，並先形成實數基底乘積。它共用原來的 immutable prepared
geometry；Green/Bessel kernels、網格、積分點、奇異修正、scatter、支承與
Schur elimination 均沿用原實作。這是浮點加總次序的改變，沒有新增核近似。

目前是可獨立重現的實驗 backend，**尚未加入正在執行的細網格建核程序**。
原程序的來源指紋與已付費頻譜保持不變。私有 function globals 只替換這四種
`einsum`；原 module、NumPy globals 與其他 contraction 均保持原樣。
這種隔離讓已固定來源的長程控制仍可繼續；日後合併正式 assembler 前仍需獨立審查。

2026-10-29 的實測使用 event-16 的完整細網格：outer 10 mm、contact 2 mm、
end/corner 0.015625 mm、Q12、531 retained inputs。取自 2 µs／45 ms Radau 計畫
（M=65536）的 slow branch 第 0、655 個實際頻率。每個頻率做兩次，順序依次為
original→factored、factored→original，BLAS 單執行緒；當時的 FEM 暫停後才開始，
完成後恢復。這是組裝實驗，不是四核心完整事件速度比。

|實際頻率 s（1/s）|原雙 body 組裝中位數|新組裝中位數|耗時減少|
|---|---:|---:|---:|
|175.6733 + 0i|54.220 s|48.212 s|11.08%|
|175.7813 + 31398.6686i|99.812 s|94.612 s|5.21%|

每次重複都檢查兩個 body 的 V/H/D、凝縮阻抗與完整邊界重建；報告保存最後一次的
誤差及所有重複的耗時。兩頻率最大矩陣 relative infinity norm 差 3.33e-13，
阻抗 relative Frobenius 差 1.56e-14，邊界重建 1.37e-14。
共享幾何準備 23.74 s；全部交替執行、檢查與報告約 623.7 s。
額外 contraction workspace 的保守估計修正為每 body 448512 bytes，不是整體 RSS。
原計時報告的 301056 bytes 漏算一個暫存實數乘積；結果檔保存原 source 與 AST 比較，
確認修正只涉及統計方法，數值收縮及已量測的時間沒有改變。
完整細節與輸入／環境指紋見 [結果](factored_galerkin_results.json)。

2026-10-30 本機 12 項測試通過，另涵蓋：

- 隨機複數 tensor 的四種收縮與原運算比較；benchmark 明確拒絕非有限矩陣／範數，
  避免 NaN 比較意外放行。既有實測誤差均為有限值，原 benchmark source 可從記錄的 commit 取得。
- 靜態、低頻、振盪及大實部頻率的完整 body operators／凝縮／重建。
- 幾何共享與不可變條件，四個並行唯讀呼叫的確定性，原 NumPy globals 不變。
- 小型真實兩 block、Q8 的完整 Radau 計畫：兩個 physical branches、全部 17 個
  half-circle 節點、32-point finite DFT、10 個因果 lags，保留 lag-zero shift wrap。
  原／新完整權重、因果應用、完整未來轉置均通過 relative norm 1e-10；
  轉置內積相符到 1e-12。這不是非線性摩擦軌跡或 initialized contact AD 驗證。

```sh
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -B bem/tools/benchmark_factored_galerkin.py \
  bem/src/finiteblock_bem/resources/cases/pmma_event16.toml \
  --output FRESH_BENCHMARK_FOLDER --indices 0 655 --branches 0 --repeats 2
```

兩個頻率不足以估計完整 contour 的效能，fast branch 尚未做細網格計時。
下一步需在新建、獨立來源的 kernel 上檢查完整非線性軌跡、原始因果方程與
initialized JVP/VJP，才可用於精度／穩定性結論。此局部改善尚未證明優於 FEM。
