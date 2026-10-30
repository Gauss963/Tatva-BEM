# 頻率局部 Bessel 函數表控制

原[細網格計時拆解](hierarchical_galerkin.md)顯示，主要複頻的 K0/K1 評估占組裝約 64%。
本原型只替換這項評估；完整原始 V/H/D、原 quadrature／singularity subtraction、
小參數差分級數、global V solve 與 Schur condensation 均保留。
尚未替換正在執行的原 producer，也沒有取得完整 FEM／BEM 速度或穩定性優勢。

## 方法與精度限制

相同 s 的 P/S 引數 `z=s*r/c` 位於同一條 right-half-plane ray。
對 `kve(nu,z)=exp(z)*K_nu(z)` 建立 dyadic radius intervals，
每段以 degree-24 Chebyshev 近似 order 0、1，再乘回**原引數**的 `exp(-z)`。
這個 scaling 的定義見 [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kve.html)；
[DLMF 10.40.2](https://dlmf.nist.gov/10.40.E2) 的大引數形式也說明，
移除指數後留下較緩變的因子。這是本專案的徑向插值實驗，並非 2026 ACA／FMM 論文的逐式移植。

每頻率的表格從 SciPy `kve` 值重新建立，兩個 order 均需通過獨立 off-grid nodes、
dyadic seams 與末四個係數 tail 的 `5e-14` 相對門檻。
不合格的整個 bin、非該 ray、超出建表區間或不支援的 order／實數型輸入，
均直接回到原 `kv`。沒有半徑 clipping 或將失敗值換成零。
這些是 **empirical checks**，沒有聲稱 uniform interpolation error certificate。

插值點準確並不自動保證徑向差分、V/H/D 或凝縮解準確，因此接受門檻仍是：

- 每個 body 的三個完整原始矩陣：`||A-Aref||_F <= 1e-14 + 1e-11*||Aref||_F`。
- 凝縮阻抗及完整邊界重建：relative Frobenius／infinity norm 均不超過 `1e-8`。
- 小型完整 Radau contour 的所有 causal weights、完整 forward history 與 future transpose，
  相對 L2 差不超過 `1e-9`，並另核對 forward／transpose duality。

表格與原 quadrature geometry 均唯讀；每次組裝用私有 function globals，
沒有改寫 SciPy、NumPy 或原 module bindings，也沒有共享可變的 last-call cache。
Clenshaw evaluation 沒有配置 degree-by-all-points 的大型係數副本。
目前不壓縮矩陣或完整時間歷史。

## 實際細網格結果

相同 531-input event-16 mesh，兩 body 各自組出完整矩陣及全邊界重建。
下表為兩次交替控制的中位數，函數表每次重建的成本也已計入：

|Radau 實際 s（1/s）|原雙 body 組裝|函數表＋組裝|耗時變化|
|---|---:|---:|---:|
|175.6733 + 0i|56.083 s|57.872 s|增加 3.19%|
|175.7813 + 31398.6686i|107.281 s|53.990 s|減少 49.67%|
|2999473.0418 + 0i|56.138 s|52.554 s|減少 6.39%|

全部六次 raw／Schur controls 通過。最大 raw normalized mixed error 為 **0.001770**，
門檻仍為 1；兩個 Schur outputs 的最大 relative infinity norm 差 **2.81e-13**。
每個 body、每個 frequency 的 30 bins 均通過 sampled checks，每表約 24 KiB。
原幾何準備 23.325 s，額外 radius-domain scan 0.418 s；
完整數值控制連獨立原矩陣與重複計時為 800.981 s，外層連當時 15 項測試為 807.722 s。
初步 14 項測試另耗 1.603 s，最後 17 項測試及來源／prefix 核對另耗 8.002 s，成本均有保存。

最後 **17 項測試通過**，包括 fresh complex points、dyadic seams、underflow-scale arguments、
exact fallback、不可用 scaled-library bin 的有效 JSON、唯讀共用、並行組裝、
兩 body 原始矩陣／Schur，以及完整小型 Radau causal history／future transpose。
fine benchmark 後只將 statistics 的非有限檢查值改成 JSON null，
數值 AST 已核對不變，並保留當時計時 source snapshot 與 SHA。

原長程 producer 的 **76 個 frozen inputs 未改**；已保存的 1,154,912,256-byte prefix
仍為 SHA256 `f3802c5c698bed22e55be770bb4672ee9ec86d50c9ab24f1e9f1fc982c905801`。
控制時先暫停原 producer／quota controller，結束後核對身分恢復；
這些暫停仍計入原長程 wall cost，不從成本中扣除。
[完整逐項結果、來源與控制成本](ray_bessel_galerkin_results.json)可供核對。

主要複頻的改善成立，但三個取樣頻率不能代表完整 contour 的加權成本。
低實頻已出現退步；尚未選定或驗證自動 hybrid policy，亦沒有完整事件的速度結論。

## 重現

```sh
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -B bem/tools/benchmark_ray_bessel.py \
  bem/src/finiteblock_bem/resources/cases/pmma_event16.toml \
  --output bem/outputs/ray-bessel-fresh --include-fast-zero
```

需先讓其他本機 numerical jobs 釋放 CPU／記憶體，避免把併行競爭當成速度差。
預設為原 2 µs／45 ms Radau plan、outer 10 mm、contact 2 mm、end/corner 0.015625 mm、Q12。
每個頻率交替兩次 original→ray／ray→original，兩 body 順序組裝，BLAS 單執行緒。
每一次近似組裝都計入建表成本；原幾何準備與額外 radius-domain scan 另行保存。
失敗會保留完整 operator arrays 與當次成本，不因 condensed outputs 通過而放寬 raw gate。

## AD 與後續門檻

這個原型沒有新增材料／幾何 AD。現有摩擦與 actuator 的 implicit AD 以固定 bulk
operator 為前提；小型 CQ transpose 控制只驗證其線性歷史環節，
不是整個 initialized nonlinear contact gradient 的驗證。

若後續加入材料導數，不能把隨材料變化的表格係數當作固定常數。
可研究由 [DLMF 10.29](https://dlmf.nist.gov/10.29) 的 Bessel 導數關係及
原線性平衡方程，建立局部 kernel JVP 與 implicit solve VJP，避免反傳所有建表／消去步驟；
仍須與原 SciPy kernel 的獨立 directional derivative、完整初始化及接觸歷史對照。
本次尚未實作或宣稱這些導數已正確。

完整事件採用前仍需跨 contour 控制、完整 nonlinear primal／initialized AD、
同誤差下的 cold kernel 與 forward／gradient 總時間，並處理
[FEM 破裂後時間收斂失敗](fem_postrupture_timestep.md)。
