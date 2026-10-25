# 重複距離的精確 Green kernel 重用實驗

2026-10-25。目的為降低接觸網格加密的冷建核成本。
已量測實際fine2/Q12的五個單頻率，完整冷建核效能仍未量測。
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

獨立FEM計時結束後，在本機依序執行量測，例如：

```sh
FDBEM_CORES=1 PYTHONPATH=bem/src python -B bem/tools/unique_radius_experiment.py \
  bem/outputs/galerkin-fine2-q12-contact16-recovered \
  --angles 0 .1 .5 2 3.141592653589793 --repeats 2 \
  --max-radial-workspace-mb 512 --output bem/outputs/unique-radius-fine2
```

工具使用來源kernel實際的dt／FFT circle，核對來源、輸入和package指紋。
結果需分開記錄索引準備成本、各頻率交替計時與完整kernel成本。
實際五個contour角度0、0.1、0.5、2、pi，各交替兩次，
完整transfer的相對Frobenius差皆為0；median時間如下。

|角度|reference [s]|unique radius [s]|時間比|
|---|---:|---:|---:|
|0|14.696|14.264|1.030|
|0.1|25.350|22.560|1.124|
|0.5|18.707|17.216|1.087|
|2|16.196|15.208|1.065|
|pi|16.070|15.083|1.065|

這是單frequency、BLAS一thread的比較，沒有同時執行其他已知求解工作。
輸入及兩套assembler的合計前置準備13.176s，不能當作單獨indices成本。
兩body的徑向點數只縮減1.289／1.348倍，因此完整算子改善有限；
目前不能把此方法當作大幅降低冷建核成本的證據。
完整數據、環境與來源見[量測紀錄](unique_radius_results.json)。
預設正式backend維持不變；與FEM的最大滑移差仍約13.2%。

## 可選的完整建核入口

`native-resolvent-kernel`新增`--assembly unique-radius`，可將同一精確r重用
接到fresh native頻率取樣、完整時間權重與因果elastic startup。
預設仍為`reference`；此功能尚未代表大型kernel已量測加速。

```sh
FDBEM_CORES=4 bem/run.sh native-resolvent-kernel SOURCE_RUN \
  --assembly unique-radius --radial-chunk-size 4096 \
  --max-radial-workspace-mb 1024 --workers 4 \
  --dt 2e-6 --duration .045 --correction-duration .025 \
  --max-memory-mb 12288 --max-spectrum-mb 8192 --output NEW_KERNEL
```

`--max-memory-mb`的陣列估計會在準備geometry之前加入shared radius indices、
排序暫存及每個frequency worker的七個徑向table。
前置估計以實際Gauss規則的點數上界計算，不假設有任何重複距離；
準備後還會核對實際indices/table尺寸沒有超過上界。
估計分開處理sorting、sampling、row IFFT和elastic startup：
sorting與worker table不同時存活，indices持續到IFFT結束，
assembler則在完整elastic startup之前釋放。這仍不是整個程序RSS的硬上限。

kernel與native spectrum的provenance保存選用方式、兩個body的unique統計及工具SHA。
新測試以真實有限雙block、每個CQ節點重新組裝的reference矩陣，
分別核對`reference`與`unique-radius`的完整權重和elastic startup；
網格覆寫、streaming hashes與完整初始條件的檢查照常保留。
