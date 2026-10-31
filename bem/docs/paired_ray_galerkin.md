# K0／K1 共用評估與完整短程滑動／AD 控制

`tools/paired_ray_galerkin.py` 將既有[scaled-Bessel ray 表](ray_bessel_galerkin.md)的
K0、K1 共用座標、mask 與 `exp(-z)`，並接上[積分收縮分解](factored_galerkin.md)。
表格 degree、off-grid／tail checks、失敗 bin 的原 SciPy fallback 均保留；
原 small-argument 差分級數、Q12、singularity subtraction、全域 V solve 與 Schur 不改。
新 backend 使用私有函數副本與唯讀 geometry，沒有 mutable last-call cache 或 module patch。
表格的接受檢查仍是 empirical，沒有 uniform error certificate。

## 相同細網格的交替計時

沿用同一個 benchmark 函數與原門檻；每個頻率、兩個 body，交替兩次原版與新 backend，
每次均計入重新建表。下列為中位數，不是完整 kernel 或完整事件的總時間：

|實際 Radau s（1/s）|原雙 body 組裝|paired＋factored＋建表|耗時變化|
|---|---:|---:|---:|
|175.6733 + 0i|54.765 s|54.040 s|減少 1.32%|
|175.7813 + 31398.6686i|104.502 s|48.978 s|減少 53.13%|
|2999473.0418 + 0i|55.461 s|49.830 s|減少 10.15%|

低實頻只改善約 1.3%，兩次樣本不足以確認其統計顯著性。
這也不是與前一版單獨 ray 表的同期對照，不能將兩輪時間差歸因於 pairing。
三個取樣頻率不能代表完整 contour 的加權成本。

六組完整 raw／Schur controls 均通過：

- raw V/H/D 原門檻為 `1e-14 + 1e-11*||Aref||_F`，最大 normalized error **0.003296**，門檻 1。
- 阻抗與完整邊界重建的 relative Frobenius／infinity 門檻均為 `1e-8`，最大 infinity error **2.86e-13**。
- 原 geometry 準備 **23.981 s**，新增 domain scan **0.422 s**；完整數值控制 **768.319 s**，外層 **769.004 s**。

原型及既有 ray／factored 合計 **42 項 targeted tests 通過**，包含精確 fallback、
兩個完整 body／Schur、私有並行組裝、小型完整 causal weights、forward history 與 future transpose。
該次測試外層 17.854 s，來源與原程序身分均有核對。

## 非零滑動與初始化梯度

`tools/check_paired_ray_nonlinear.py` 另外完整組裝小型 contour 的 **514 個物理模式**，
再檢查兩個 lag-zero 模式，通過每一個模式的相同 raw／Schur 門檻。
兩個 backend 共用原 healed body preload 與其 selected active-set sensitivities；
每支獨立建立兩個完整 CQ weight families、wrap、startup，完成 128 步與 (8,16) 接觸子步。
保留 measured initial residual force 的全部歷史。

此控制使用 h/contact/end/corner = 160 mm、Q12、dt = 2 µs、FFT size 512。
致動器速率明確設為診斷用 **7812.5 mm/s**，不是原 event-16 的 20 mm/s。
末端最大累積滑移約 **1.033822 mm**，已經歷 weakening，並非全程 sticking 的控制。

|完整短程對照項目|新 backend 相對原版的 L2 差|
|---|---:|
|全部 causal weights|2.43e-14|
|所有 stage displacement|9.35e-14|
|所有 stage contact response／state／work channels|2.17e-13|
|五個摩擦參數的初始化梯度|2.20e-12|
|全部 stage actuator 梯度|3.29e-13|

兩支各自對五個摩擦參數與兩個完整 stage-loading directions 做 initialized JVP／VJP 對照，
並比較兩 backend 的 gradients／JVP；全部通過原 `rtol=1e-7, atol=1e-18` mixed gates。
這是固定 bulk／restart／已記錄 loading、所選初始化與暫態分支的整合控制，
尚未新增完整軌跡的獨立 finite-difference 檢查，亦不包含材料／幾何或 endogenous stopping-time 導數。
本次控制共 **64.534 s**；與 FEM 唯讀分析串行的外層合計 **82.961 s**。

所有暫停均保留在原長程工作 wall cost。原 BEM 的已付費 prefix 前後 SHA 相同，
控制結束已恢復相同 owner／quota／numerical 程序。
[完整結果、來源快照／hash、輸出 hashes 與成本](paired_ray_galerkin_results.json)保留原始證據。

## 執行與後續採用門檻

```sh
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -B bem/tools/benchmark_paired_ray.py \
  bem/src/finiteblock_bem/resources/cases/pmma_event16.toml \
  --output bem/outputs/paired-ray-fresh --include-fast-zero

PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
JAX_ENABLE_X64=true python -B bem/tools/check_paired_ray_nonlinear.py \
  --output bem/outputs/paired-ray-sliding-fresh
```

數值工作需與其他本機計算串行並受四核心 CPU 預算控制。
目前沒有更換正在執行的原 producer。採用前仍需更廣 contour 的成本／精度控制，
完整細網格 primal、原始因果方程與初始化梯度檢查，以及同誤差的 cold kernel／forward／gradient 總時間。
FEM [dt/16 參考收斂仍未通過](fem_dt16_window.md)，完整目標保持未完成。
