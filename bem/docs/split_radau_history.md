# 分離 Radau 啟動與接觸修正歷史

完整 [45 ms 場比較](recovered_full_event.md) 在約 29 ms 後出現壓力、剪力與滑移差異。
下一個時間控制以 **1 µs、29.4 ms** 為候選，涵蓋 FEM 已觀察到的第一次接觸張開與彈簧分離。
此視窗用於定位分歧。全程比較仍須涵蓋原本的 45 ms。

## 兩項儲存改動

在閉合黏著的基準切線下，完整啟動歷史只需要兩個實際作用力的響應。
[啟動歷史工具](../tools/projected_radau_startup.py)對單位 platen 力和原初始平衡殘力
分別計算 (J(K(s)+J^T T J)^{-1}J^T f)。(K) 是支承消去後的 BEM 剛度，
(J) 是接觸與 platen 的位移映射，(T) 是接觸與彈簧的黏著切線。
兩個響應沿完整時間輪廓保存。ramp 保留 finite-DFT 的 lag-zero wrap，
預載殘力歷史保留全部正 lag，lag zero 由目前步的平衡方程處理。
兩者皆使用原 float64 模型。

[分塊互易儲存](../tools/reciprocal_radau_storage.py)保存兩個原 Radau branch。
複數 work-coordinate 算子使用雙線性轉置 (H^T)；每個非對角分塊只存一份。
每個矩陣都檢查相對 Frobenius 轉置差不超過 (10^{-12})，
超過門檻或頻率寫入不完整就拒絕發佈。IFFT 直接由兩個分支生成全部指定 lag 的
float64 Fortran-order 權重與 lag-zero wrap。省下的是頻率矩陣的重複區塊。

較短的修正輪廓只用於第一次非零非線性修正之後。若實際修正早於計畫範圍，
完整時間步進就需要更長的權重。改變 FFT 週期也會改變有限輪廓的混疊誤差，
因此同一時間步的原算子與整段軌跡仍須獨立核對。

## 驗證

- 57 步耦合振動控制與完整 stage 矩陣直接卷積相比，ramp 相對 L2 差 (1.51	imes10^{-14})，
  預載殘力歷史相對差 (1.14	imes10^{-13})。
- 240 步振動控制將修正 FFT 從 512 縮至 64，全部時刻的相對差
  (1.03	imes10^{-11})，單看修正響應相對差 (6.09	imes10^{-10})。
  另以提早發生的非零修正測試長度拒絕。
- 雙 block 128 步粗網格案例使用原 2 µs 步長與診斷載入。
  完整啟動 FFT 為 512，重新組裝的修正 FFT 為 256。滑移都從第 41 步開始，
  階段位移與接觸資料的相對差分別為 (2.49	imes10^{-11}) 與 (3.57	imes10^{-12})。
  原接觸 residual 最大 (1.61	imes10^{-8}) MPa，低於 (10^{-7}) MPa。
- 同一粗網格的 257 個頻率點與 129 個 lag 全數對照原頻譜與權重。
  兩個權重族的相對 L2 差分別為 (4.42	imes10^{-16}) 與 (4.88	imes10^{-16})，
  wrap 最大絕對差 (2.88	imes10^{-24})。細網格抽查 28 個原始／compact 頻譜矩陣，
  最大相對轉置差 (2.73	imes10^{-14})。

## 1 µs 的磁碟預檢

`27.8 ms` 是暫定的最早可容納修正時間。2 µs 案例的首次修正在 28.13 ms，
1 µs 仍需在時間步進驗證。29.4 ms 視窗的修正計畫有 1600 個 lag，
完整啟動 FFT 65536，修正 FFT 4096。531 個 trace DOF、分塊寬 32 時，
兩個 reciprocal 分支需 **9.12 GiB**，完整啟動與修正權重等需 **8.23 GiB**。
在本機外接與內部磁碟分開存放，分別保留 3.5 與 4 GiB 後通過資料量預檢。
該結果只計算預估容量，沒有預留磁碟或建構 1 µs 歷史。

```sh
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 python -B bem/tools/plan_split_radau_history.py KERNEL \
  --dt 1e-6 --duration .0294 --earliest-correction .0278 \
  --spectra-volume SPECTRA_DIRECTORY --time-volume TIME_DIRECTORY \
  --spectra-reserve-gib 3.5 --time-reserve-gib 4
```

目前的公開工具涵蓋歷史轉換與磁碟預檢。整合式取樣與細網格推進、
與 FEM 的時間收斂及同精度成本，以及新路徑的初始化 JAX 導數均仍待驗證。
