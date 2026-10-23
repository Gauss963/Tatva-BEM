# 彈性核組裝的等價重寫實驗

2026-10-24。此實驗針對冷建立成本，沒有更換 BEM、摩擦律或時間積分。
目前細網格與 FEM 的滑移仍差約 25%，不以這項局部加速作為完整目標已達成的證據。
[數據、執行條件及 SHA-256](radial_assembly_results.json) 保留來源。

## 少建立中間張量

現有 Green tensor 為 `G = a*I + b*e*e.T`，`e = vector/r`。
以 `c=b/r`、`da=∂a/∂r`、`db=∂b/∂r` 表示，先對源點法向 n 收縮可得

```text
alpha = mu*(da+c)
beta  = 2*mu*(db-2*c)
gamma = 2*mu*c + lambda*(da+db+c)
T_n G = alpha*((e.n)*I + n*e.T)
      + beta*(e.n)*e*e.T + gamma*e*n.T.
```

它與原本先形成 `gradient[k,i,j]`、再做 divergence／normal contractions 的式子等價。
反向 pair 使用 `-T_n G`，維持 source-minus-target 方向及原 bilinear reciprocity；
沒有把矩陣取 Hermitian part、裁剪特徵值或刪減積分點。
此式可省去每個積分點的三階梯度陣列及相關收縮暫存。

## 相同九項級數的 Horner 求值

小參數的原始 K0 差導數使用 k=1…9 的有限級數。
對 `n=2*k`、原係數 A、B，其三個導數分別含

```text
r^(n-1) * [n*(A+B*log(r))+B]
r^(n-2) * [n*(n-1)*(A+B*log(r))+(2*n-1)*B]
r^(n-3) * [n*(n-1)*(n-2)*(A+B*log(r))+(3*n^2-6*n+2)*B].
```

將其重排為 r² 的三個 Horner 多項式，再乘 r、1、1/r。
保留相同九項與原係數，不引入新的截斷或 Bessel 插值。
相同頻率跨幾何 batches 的係數以至多 32 組的 LRU 共用；資料不可變，
四個 workers 的共享幾何／冷 cache 測試得到相同矩陣。

## 核對與效能範圍

- 任意法向、不同 Laplace 頻率與跨小參數範圍的點值：對完整梯度公式核對。
- 不規則四邊形，static／dynamic：V、H、D 與 Schur stiffness 均通過比較，
  原 package 的 globals 維持同一物件。
- 原粗網格 4 us contour 的六個角度：transfer 矩陣相對 Frobenius 差最大約 1.08e-12。
- 32／64 步診斷，以新算子**重新建立 dense CQ 權重**後推進：
  滑移最大差小於 1e-14 mm；所有精確接觸分支相同。
  五參數梯度最大相對差小於 3e-12，亦核對每步 actuator 及初始 z／c cotangent。
- 另以 JAX JVP／reverse AD 核對 traction 收縮對幾何方向、徑向係數及 Lamé
  係數的局部導數。這不是 E／rho 經 Bessel、完整 bulk、lift 與事件時間的端到端梯度。

成對計時採同一粗網格／Gauss 12、單個 frequency worker、BLAS 一執行緒，
四個 contour 角度各交替執行四次。另一個重計算在這 48.10 秒內停止後接續，
避免兩者直接爭用 CPU；沒有控制其他 macOS 背景工作。
各頻率 median 的原版／原型時間比約 1.06–1.21；四個 medians 相加之比約 1.11。
這只是抽樣單頻率組裝的改善，未量測完整冷核或與 FEM 同精度的速度優勢。
原 2 us 事件核的 wall time 包含該暫停與早先診斷重疊，不可當受控效能數據。

## 隔離與重跑

原型在 `tools/radial_traction_experiment.py`，以局部 function bindings 複製現有
prepared assembler。正在執行的正式算子、原快取與 package source 不會被替換。
這是便於逐項核對的實驗方式；通過完整事件及資源檢查後，才考慮整合進正式組裝器。

```sh
FDBEM_CORES=1 PYTHONPATH=bem/src python -B bem/tools/radial_traction_experiment.py \
  SOURCE_RUN --angles 0 .1 2 3.141592653589793 --repeats 4 --output NEW_KERNEL_REPORT
FDBEM_CORES=1 PYTHONPATH=bem/src python -B bem/tools/check_radial_rollout.py \
  SOURCE_RUN --output NEW_ROLLOUT
```

依 `bem/run.sh` 設定 BLAS limits。輸出資料夾須為空；長算例先檢查記憶體預估。
完整案例還需核對新的時間權重、45 ms 非線性軌跡、分支、adjoint 及穩定性指標。
目前不切換預設 backend。
