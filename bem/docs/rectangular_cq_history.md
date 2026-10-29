# 有界記憶體的內部觀測 CQ 與 JAX VJP

2026-10-29 新增 [`rectangular_cq_history.py`](../tools/rectangular_cq_history.py)，
把全部 retained stages 映射到內部場量，亦提供完整未來依賴的實數轉置。
它逐批處理 output rows，不保存完整時間權重或展開的二階段矩陣。
[`reconstruct_interior_reference.py`](../tools/reconstruct_interior_reference.py)
已用實際雙 block 的完整短程軌跡重建 12 個固定實體 probes。

driver 保留有預算上限的直接頻率取樣，另已接入
[可續跑的 body／probe raw producer](interior_raw_sampling.md)及 `--prepared-kernel`。
完整 45 ms 的建核與場量尚未執行；完整 FEM 波場、initialized AD 與
同精度效能目標仍未完成。既有 FEM／2 µs 排程保持原狀，原 Tatva 維持只讀。

## 因果歷史與檔案生命週期

輸入是 physical retained perturbation stages，人工初始 block 固定為零。
對 `I*F0(z) + Delta(z)*F1(z)`，每批先 IFFT 得到**全部 requested physical lags**，
再對有限因果權重作 zero-padded FFT；長度至少為 `2*(steps+1)`。
保留 `Delta1` shift 與有限 DFT lag-zero wrap，不引入未來 source 或刪除反射歷史。
直接將 contour samples 乘上有限 source FFT 會引入未來輸入，測試明確顯示此差異。
反向對同一有限因果映射作 correlation、空間／stage 轉置與完整時間累加。
人工初始 block 的 cotangent 固定為零，不需要保存 forward trajectory。

`FixedRectangularCQ` 初次以串流 SHA256 核對完整 complex128 NPY 頻譜，
並檢查核對期間 inode／size／mtime 未變。直接 apply 前後再核對 identity；
source maps 只讀，平台支援時會在 copied stripe 之後釋放 mapped cache。
array workspace guard 不含 caller state、BEM assembly、JAX buffers 或 OS page cache，
不是 process RSS 上限。

## JAX 與導數範圍

`as_jax()` 提供固定線性映射的 custom VJP，可在 `jit` 內組合目標函數；
`vmap` 使用 sequential callbacks，避免同時複製多份 FFT workspace。
pullback 回傳全部實際 input stages 的 cotangents，不儲存整份 forward trajectory。
FFT 在 CPU callback 中執行，尚未量測大型效能，也沒有 GPU 加速的宣稱。

檔案須在 operator lifetime 保持 immutable，object 必須保持開啟。
在 compiled region 外 `verify()`，等待 JAX 結果完成後再核對一次。
callback 只做固定核心數值運算，不核對檔案或寫入；JAX 可重排／省略 pure callbacks，
因此不能依 callback 保證 provenance checks 執行。輸入須為有限 float64、固定零人工起點。
參考 [JAX callbacks 文件](https://docs.jax.dev/en/latest/201/callbacks.html)。

這是對 **retained perturbation stages 的一階 reverse derivative**。
`.apply(direction)` 可作獨立線性 JVP 核對；custom VJP 不提供 JAX forward-mode JVP。
初始化 lift、材料／幾何、kernel、stop-event time 與完整 contact solver 導數須另外接回。
float64 依 [JAX X64 文件](https://docs.jax.dev/en/latest/101/default_dtypes.html)；
第一次測試使用了本機已移除的舊 context API，失敗與成本保留，已改用現行 API。

## 實際雙 block 控制

driver 核對 source kernel／case／seed／runtime、完整 stage clocks 與檔案指紋。
`Tq(s)` 使用 physical 座標，沒有再套 work normalization。
靜態 lift 從原完整 P1 boundary displacement 與 `V lambda = H u` 恢復；
動態使用完整 CQ，velocity 為 `s*Tq(s)`，靜態 velocity 為零。
原 q 與量得的 initial residual force 均不修改或清零。
每點輸出 `ux,uy,vx,vy,sxx,syy,sxy,szz`，先 moving、再 stationary，保留兩個 stage 的時鐘。
FEM velocity 比較仍須使用原 half-step clock。

[完整結果與指紋](rectangular_cq_history_results.json)：

- 9 項新測試與 23 項原 interior controls，**32 項通過**。
  包括不同 rectangular shapes／rows、dense 獨立 IFFT、完整轉置／內積、未來輸入洩漏、
  資料損毀／初始核對競態、preflight、JIT VJP／vmap 與實際 BEM → contact → interior driver。
- 額外控制為 Q12、11 unknowns、2 µs、16 步、FFT 64、原 f09 seed，
  12 probes／96 output channels，全部 stages 重建，沒有擬合相位。
- 全部增量相對獨立 dense CQ 的 L2 差為 `8.29e-15`；
  完整 transpose directional error 為 `9.14e-15`；JAX fixed-map VJP 差為零。
- 原靜態位移／應力加回，初始 stage 為原靜態場、velocity 為零。
  此短程來源維持 sticking，未驗證完整 slip-weakening 或反射波 FEM。

所有本機 numerical controls 一核心依序執行，期間暫停並恢復同一 FEM／quota。
失敗與重核對成本均保存；等待已在 FEM wall interval 中，不可再加一次。

## 完整範圍的邏輯估算與使用

531 inputs、12 probes／96 outputs、每批一列：

| 45 ms 時間網格 | forward arrays | transpose arrays | 原 dense estimate | compact spectra data |
| --- | ---: | ---: | ---: | ---: |
| 2 µs | 2404.3 MiB | 2499.3 MiB | 340875.6 MiB | 49.783 GiB |
| 1 µs | 4744.6 MiB | 4934.6 MiB | 681734.0 MiB | 99.564 GiB |

這些是邏輯估算，尚未執行完整細網格；不含其他核心、FEM、完整原方程稽核或 caller state。
磁碟共存與建核仍需安排，不能用估算宣稱完整事件更快／更穩定。
對保存原 static lift 的完整短程 run，可執行：

```sh
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_ENABLE_X64=true \
python -B bem/tools/reconstruct_interior_reference.py PATH_TO_COMPLETE_RUN \
  --output bem/outputs/new-interior-reference --output-rows 1 \
  --max-workspace-mb 512 --dense-check --check-ad
```

直接 sampler 預設最多 256 次 physical-frequency evaluations；超過即拒絕，不縮短 source horizon。
完整 event 尚需執行 body／probe sampling、true-float64 FEM probes、
空間／時間／初始化梯度收斂、能量與同精度總耗時／穩定性驗證。
