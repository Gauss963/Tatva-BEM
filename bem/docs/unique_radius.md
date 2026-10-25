# 重複距離的精確 Green kernel 重用實驗

2026-10-25。目的為降低接觸網格加密的冷建核成本，尚未量測完整案例效能。
此實驗保持原P1/P0 Galerkin積分、奇異修正、source traction與完整歷史。

目前Laplace Green kernel的七個徑向係數只依賴r、s及材料；
方向、法向與基底收縮仍在各自的積分點計算。
`unique_radius_experiment.py`事先對**浮點數完全相等**的r建立索引，
每個頻率只算一次各unique r的Bessel函數，再逐batch取回原係數。
沒有rounding、徑向插值或省略積分點；每次取回還核對實際r與預備索引完全相同。
這是本專案的代數重用實驗，不是論文中3D-ACA或空間FMM的實作。

新增的radial table在每次assemble內建立，workers只共享唯讀geometry／indices。
顯式budget限制額外radial workspace，沒有把它當成整個assembler的RSS上界；
`np.unique`的排序暫存、原geometry與矩陣記憶體仍須另外評估。
靜態分支直接沿用原式。稠密線性求解與矩陣收縮沒有加速保證，
因此unique數量減少不能直接當作整體速度倍數。

回歸測試比較static與跨小參數／振盪頻率的完整V/H/D矩陣，
並檢查真實雙block的condensation、四workers一致性、globals及workspace限制。
測試與大型kernel／45ms軌跡／完整adjoint驗證分開列出。
此路徑也沒有提供材料或幾何AD；geometry索引在這個實驗中固定。

待獨立FEM計時結束後，才在本機執行量測，例如：

```sh
FDBEM_CORES=1 PYTHONPATH=bem/src python -B bem/tools/unique_radius_experiment.py \
  bem/outputs/galerkin-fine2-q12-contact16-recovered \
  --angles 0 .1 .5 2 3.141592653589793 --repeats 2 \
  --max-radial-workspace-mb 512 --output bem/outputs/unique-radius-fine2
```

工具使用來源kernel實際的dt／FFT circle，核對來源、輸入和package指紋。
結果需分開記錄索引準備成本、各頻率交替計時與完整kernel成本。
預設正式backend維持不變；目前與FEM的最大滑移差仍約13.2%。
