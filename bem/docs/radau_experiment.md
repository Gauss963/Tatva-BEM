# Radau IIA 時間積分研究

2026-10-24。這是放在 `tools/radau_experiment.py` 的獨立原型，沒有替換正式 BDF2。
目標是降低時間色散／耗散，讓有限 block 的破裂能以較少步數解析。
**尚未以 Radau 完成原 event-16，也未證明同精度速度或整體穩定性優勢。**

## 方法與閱讀依據

[Kramer、Marussig、Schanz（2026）](https://arxiv.org/html/2510.06804v1)
結合高階空間基底與 RK-CQ，並討論 mixed boundary problems 的時間／空間誤差。
本原型先保留目前 P1/P0 空間離散，測試兩階段 Radau IIA，沒有移植 IGA。

[Banjai、Lubich（2019；2017 預印本）](https://arxiv.org/html/1702.08385)
給出 RK convolution coercivity 與非線性波問題的分析。其非線性收斂結果要求
strict monotonicity 和足夠正則性，不能直接套到 slip weakening。

採 stage-major 排列，對每個 FFT 圓上的 z，計算

```text
A = [[5/12, -1/12], [3/4, 1/4]], b = [3/4, 1/4], c = [1/3, 1]
Delta(z) = inv(A) - z * outer(inv(A) @ ones(2), [0, 1])
Z_RK(z) = Z(Delta(z)/dt).
```

以二階矩陣的特徵分解評估矩陣函數；接近缺陷特徵基時明確拒絕。
兩個 trial stages 共用上一步已接受的塑性／累積滑移，只提交最後一階段狀態。
這保留 previous-state 係數的更新方式，但沒有因此宣稱接觸歷程具有三階精度。

## 已完成的結果與反例

先以精確有限桿 transfer `tanh(s)/s` 排除空間誤差。同一半餘弦載入及反射區間，
載入端位移相對 L2 誤差如下；載入的正則性也不符合任意高階收斂假設。

|dt|BDF2|Radau IIA|
|---|---:|---:|
|0.08|2.2513%|0.2070%|
|0.04|1.0010%|0.04132%|
|0.02|0.3771%|0.006641%|
|0.01|0.1111%|0.001026%|

雙有限 block 的短程診斷使用原材料／摩擦律與 h=80 mm，但把 actuator 在
256 us 內增加 2 mm，沒有使用原事件的 20 mm/s 載入。下表不是 FEM 驗收。

|dt|Gauss|BDF2 最大 c [mm]|Radau 最大 c [mm]|
|---|---:|---:|---:|
|8 us|12|0.906945|0.929233|
|4 us|12|0.925048|0.939445|
|2 us|12|0.937878|0.945143|
|1 us|12|0.945403|0.949016|
|0.5 us|12|0.949225|0.758188|
|0.5 us|24|0.949192|0.951058|

最後兩列揭露重要的積分敏感度：Gauss 12 的 Radau 結果不能列為收斂成功，
即使其 Newton 殘差約 1.4e-11 MPa、累積滑移非遞減。提高 Gauss 後結果恢復接近
前一組趨勢，但 Gauss 24 仍須獨立加密，不能直接指定為真值。

另以獨立的耦合 ODE 標準 RK 離散，核對全部 stages 的 CQ impedance 與 compliance；
並檢查多項式 composition 與 b 加權 coercivity。JAX 使用現有局部切線與隱式根微分，
短程 direct-history scan 只供 AD 核對，尚未實作可擴展的 RK reverse-CQ。
1 us／256 步的 mu_s、mu_k、Dc 差分相對差約 5.4e-7、2.0e-7、3.3e-6，
兩個 stage 的開閉、滑動與 weakening clip predicates 都未變更。

目前實際動態runner的 stage 矩陣儲存量為單階段的四倍，每個 FFT 角度需兩次 Laplace 評估。
因此波傳誤差下降本身不足以證明總時間優勢，後續仍需處理頻譜／歷史成本。
程式沒有移除歷史尾端，也沒有額外加入物理阻尼。

## 真實Q12算子核對與兩個weight families

2026-10-25的[頻譜重用](cauchy_reuse.md)核對已涵蓋8us Radau的五個完整
550×550 stage matrices、每點兩個重新計算的275×275 BEM modes。
最大相對差1.68e−11，通過原1e−6門檻；stage eigenvector條件數不超過1.620。
完整頻率映射的8us目標通過原解析域guard；4us仍有目標越界，沒有放寬guard。
這只是抽樣算子檢查，尚未建立完整Radau核心或完成原事件。

另以Cayley–Hamilton恆等式推導可選的精確儲存表示。
對dimensionless的兩階段Delta及其eigenvalues lambda0、lambda1，令
Lj=L(lambdaj/dt)，則

```text
F0 = (lambda1*L0 - lambda0*L1) / (lambda1-lambda0)
F1 = (L1-L0) / (lambda1-lambda0)
L_RK(z) = I*F0(z) + Delta(z)*F1(z)
Delta(z) = Delta0 + z*Delta1.
```

兩個d×d coefficient families即可表達四個stage blocks，沒有做低秩截斷。
時間權重的z*F1項為一個lag位移；有限damped DFT的lag零還有
`radius * IFFT(F1)[-1]`，必須保留，不能把它靜默刪除當作causal tail。
`compact_matrix_function`、`invert_compact_spectrum`與小型reference expansion
在`radau_experiment.py`，以獨立coupled ODE stage solve、完整dense RK-CQ權重、
有顯著aliasing的manufactured circle及全部causal輸出核對。
這是兩family實作的代數驗證；尚未接到磁碟producer、完整非線性runner或adjoint，
也沒有將理論上減半的矩陣元素數列為已量測的記憶體／速度優勢。

## 重跑

從 repository 根目錄執行，輸出留在 `bem/outputs`：

```sh
FDBEM_CORES=1 bem/run.sh radau-experiment --output bem/outputs/radau-analytic-current
FDBEM_CORES=1 bem/run.sh radau-experiment --bem --output bem/outputs/radau-bem-contact-current
FDBEM_CORES=1 bem/run.sh radau-experiment --bem --refinements 8 16 \
  --output bem/outputs/radau-bem-contact-finer
FDBEM_CORES=4 bem/run.sh radau-experiment --bem --refinements 16 --quadrature 24 \
  --workers 4 --output bem/outputs/radau-bem-contact-q24
FDBEM_CORES=1 bem/run.sh radau-ad-check bem/outputs/radau-bem-contact-finer --refinement 8
```

完整紀錄含程式／環境指紋，整理於 [radau_results.json](radau_results.json)。
這些探索性 runs 與其他工作重疊，耗時不用作公平速度比較。
