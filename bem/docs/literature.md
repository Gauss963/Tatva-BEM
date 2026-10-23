# 近期文獻與 FEM 對照的實作決策

檢索／閱讀日期：2026-10-23。範圍是有限彈性體的 fully dynamic 接觸、
時間域 BEM、卷積歷史加速與角落誤差；以下不是整個 BEM 領域的完整系統性回顧。
目標是在相同物理模型與可量化誤差下重現 FEM，然後證明穩定性或總時間的優勢。
目前尚未達成這個目標。

## 直接相關的研究

|文獻與閱讀範圍|對本模型的決策|
|---|---|
|Aimi, Di Credico, Gimperlein, **Space–time boundary elements for frictional contact in elastodynamics**, CMAME 427 (2024), 117066。[論文／全文](https://arxiv.org/html/2405.08566v1)，[DOI](https://doi.org/10.1016/j.cma.2024.117066)。已讀兩物體接觸、算子、離散、求解與數值實驗。|以 energetic space-time Galerkin 與對稱 Poincaré–Steklov 算子作為穩定性升級方向。其嚴格誤差結果針對 unilateral Tresca；Coulomb 和兩物體結果含數值證據，不能直接宣稱涵蓋目前的非單調 slip weakening。|
|Melenk, Nick, **Parsimonious convolution quadrature**, arXiv:2410.15079v1 (2024) 預印本。[全文](https://arxiv.org/html/2410.15079v1)。已讀分段插值、BDF2 延伸與實作章節。|先減少重複 Laplace 求解，保留 dense CQ 對照。採自適應 Chebyshev 版本，另加獨立點檢查及直接計算 fallback；這不是原文預設分割的逐式重現，也不沿用其複雜度／誤差證明。|
|Kramer, Marussig, Schanz, **A higher-order time-domain boundary element formulation based on isogeometric analysis and the convolution quadrature method**, CMAME 451 (2026), 118609；2025 年預印本。[全文](https://arxiv.org/html/2510.06804v1)，[出版資料](https://tugraz.elsevierpure.com/en/publications/a-higher-order-time-domain-boundary-element-formulation-based-on-/)。已讀混合算子、空間基底與收斂實驗。|P1／更高階空間離散及 Radau IIA 值得實作，但平滑波的高階收斂不能直接套到接觸開閉、yield 與 weakening 切換。先分離空間和時間誤差，不直接放大 dt。|
|Keshava, Schanz, **A 3D-ACA accelerated time domain boundary element method for elastodynamics using FMM and H-matrix techniques**, 2026 預印本。[全文](https://arxiv.org/html/2607.15292v1)。已讀 gCQ、face/fiber 壓縮及數值比較。|後續以頻率／空間張量壓縮解除 fine-mesh 歷史儲存瓶頸。其 3D 線性案例不是 2D 摩擦驗證；FMM 也不保證在本機四核心比 H-matrix 快。|
|Schanz, Keshava, **Comparison of H-matrix- and FMM-based 3D-ACA for a time-domain boundary element method**, Computational Mechanics 78 (2026), 563–578，2026-03-25 出版。[全文](https://link.springer.com/article/10.1007/s00466-026-02768-9)。|這篇是 acoustic wave／gCQ 的壓縮比較，與上面的 elastodynamics 預印本分開看待。借鑑其誤差與成本比較方式，不把聲學結果當成本問題的證明。|
|Aimi, Desiderio, Gimperlein, Guardasoni, **Space-Time Boundary Elements on Graded Meshes for 3D Elastodynamics**, IJNME 126 (2025), e70163。[出版全文](https://onlinelibrary.wiley.com/doi/10.1002/nme.70163)。閱讀重點為奇異性、grading、時間步與積分誤差的適用條件。|角落加密須配合鄰接面與積分精度；單看最小面板長度或成核時間不足以判斷收斂。其 3D Galerkin 的 large-CFL 結果不能替目前 P0 collocation 的 8 µs 精度背書。|

另檢索到 2026 年 mCCSR 平行 TDBEM 工作，但本次沒有取得可讀全文，
因此未作為程式推導或速度結論的依據。高階 p-version CQ 的 SIAM 2025
[出版資料](https://epubs.siam.org/doi/10.1137/24M1642524) 亦列為後續閱讀，
目前不據此宣稱對不光滑摩擦歷程有指數收斂。

## 從文獻到目前程式

原 backend 使用 `V^-1(1/2 I+K)` 的 P0 collocation。
它在連續層級與能量相關的表示等價，但離散後不能把兩者視為同一方法。
把數值矩陣直接取對稱或裁掉負特徵值，也不能替代推導 Galerkin 式。

本輪先加入可重現的線性 bulk 檢查：力與位移的 transfer 是 `Z(s)`，
對速度的 impedance 則是 `Z(s)/s`。在 `Re(s)>0` 檢查它的 Hermitian part、
互易性及負能量方向。這是我們對離散算子的必要條件檢查：
找到負方向可以否定該取樣點的 passivity；通過有限取樣不能證明全域穩定，
更不能證明非單調摩擦問題的穩定性。程式不會用這個檢查改寫物理解。

加速部分新增 `--sampling adaptive`，目前只支援 interface formulation。
它在 FFT 角度上逐段擬合完整廣義力矩陣，用 traction 單位縮放檢查誤差；
同時檢查 Chebyshev 係數尾端與未參與擬合的點。不能通過就細分，
細到直接計算較省時時退回原 FFT 網格。dense sampling 仍是預設。
這只減少核心評估，沒有截短時間記憶，也沒有實作 3D-ACA 或 oblivious CQ。
因為插值點檢查不是嚴格的 operator-norm 上界，新解析度仍須核對完整軌跡。

記憶體預估同步改為按實際配置分階段計數：半頻譜加分列 IFFT，
或歷史權重加完整 dyadic FFT cache，再加空間核心與 workers。
另計 SciPy 對不足長度的 kernel FFT 做零填補時，與快取並存的實數暫存。
不再把早期完整複數 IFFT 的已移除配置重複計入。這不是歷史壓縮。

## 本輪量測與重要發現

完整資料、版本與程式指紋見 [literature_results.json](literature_results.json)。

同一個角落加密 45 ms 案例，新的 sampling 把 Laplace 評估由 8193 次降至
4196 次；權重相對 Frobenius 差為 3.34e-13，完整累積滑移最大差約
2.19e-12 mm，滑動分支未改變。這是對原 BEM 離散解的一致性檢查，
並未修正其約 39% 的 FEM 最大滑移偏差。探索性冷建立約 534 s，期間與短暫的
單執行緒 passivity 檢查重疊；它不與舊的 773 s 直接構成公平速度比。

另以相同幾何、時間網格、4 workers、BLAS=1、無權重快取做獨立對照，
並交換 dense／adaptive 執行順序。此小型核心建立 benchmark 與完整
45 ms 非線性對照分開記錄，不能當作 FEM 同精度的總速度結論。

|執行順序|dense 核心建立|adaptive 核心建立|耗時減少|
|---|---:|---:|---:|
|dense → adaptive|37.16 s|33.28 s|10.4%|
|adaptive → dense|37.99 s|31.92 s|16.0%|

此表是 2048 步、均勻上限 40 mm、相同 dt=8 µs 的兩次量測，
不是統計信賴區間。CPU 時間亦有下降，但評估次數減少不會等比例轉成總速度。

更重要的是，三組空間網格的 bulk velocity impedance 在部分取樣頻率
出現負的 Hermitian 方向。接觸面 5 mm 的例子，`s=200+2*pi*5000i`，
12／24／48 點積分得到的最小特徵值分別約為
−7.00140576e-4／−7.00140550e-4／−7.00140547e-4
（採程式中的 work-conjugate 正規化）。完整邊界重建所得的功率與其一致。
因此這個取樣失敗不能由提高 Gauss 階數排除，也不只存在於最粗的接觸面。

![線性 bulk 被動性與互易性檢查](figures/passivity_audit.png)

此結果沒有證明目前特定載入軌跡必然發散，卻已排除對這些離散算子作
無條件被動性宣稱。穩定性路線因此優先轉向一致的 P1／Galerkin 能量形式，
而不是僅加密 P0 後就宣告完成。新的 sampling 可繼續服務於兩種離散的成本控制。

後續已實作 [P1/P0 對稱 Galerkin 與 nodal-force 接觸](galerkin.md)，
接回完整 CQ 與 JAX adjoint。hypersingular regularization 另依據
[Bao、Xu、Yin (2017)](https://arxiv.org/html/1608.08759v1)，
以獨立 traction 有限差分核對。此頁的 P0 數值保留作歷史紀錄；
新方法的被動性、解析波、組裝效能與 FEM 差異記錄於新報告。

完整 reverse-CQ adjoint 也已重跑：相對 dense CQ 的五參數梯度最大絕對差
約 5e-10；mu_s、mu_k、Dc 的小步長差分相對差均低於 1e-5。
依然只涵蓋固定 bulk、初始場和保存載入歷程的條件式導數。

接觸 5 mm、角落 2 mm、45 ms／8 µs 的修正 preflight 預估約 10036 MiB，
目前只有資源規劃，沒有將它列為已完成或已驗證的破裂模擬。
本輪本機 49 個 BEM tests、Ruff、formatting 及 wheel 建置通過。

## 一致性與效能驗收

第一輪工程門檻暫定為：成核與停止載入時間偏差不超過 2%；最大滑移、
末界面滑移分布與剪力歷程的相對差不超過 5%，且 BEM 自身網格／時間加密的
差異應低於與 FEM 的允許差。壓力、platen、破裂前緣與波反射也要一起檢查。
這些是本專案的驗收選擇，不是引用論文給出的誤差界；達標後仍需看局部異常。

在不改 E、nu、rho、mu_s、mu_k、Dc、penalties 與載入政策的前提下，依序處理：

1. 比對初始界面 traction／塑性狀態，量化 static lift 與 FEM 初始內部場的差異。
2. 同時解析接觸面 process zone、端點與相鄰角落，獨立縮小 dt；不以成功收斂的 Newton 代替物理精度。
3. 對照完整 dense CQ 驗證加速誤差、接觸分支、停止事件與 JAX 梯度，再用加速版本跑加密研究。
4. 若取樣 passivity 或空間收斂不合格，優先實作一致的 P1／Galerkin 能量形式；不靠額外阻尼掩蓋。
5. fine-mesh 仍受權重儲存限制時，才接續低秩／分塊歷史壓縮及其轉置 adjoint。

速度需在同精度、同四核心設定下分開記錄冷核心、初始化、JIT、推進與輸出，
並另列同幾何摩擦校準的快取成本。穩定性需檢查殘差、耗散、能量異常與
dt 敏感度；物理上的 slip-weakening 破裂不能被當成要數值消除的「不穩定」。

JAX 的策略維持固定 bulk 重用、局部 VJP、平衡根隱式微分及完整 reverse-CQ。
目前 adaptive 選點與 SciPy bulk 並未對材料／幾何可微；新增加速不會自動補上
初始場與載入停止事件的導數。

## 重跑

```sh
FDBEM_CORES=4 bem/run.sh passivity
FDBEM_CORES=4 bem/run.sh event16 --formulation interface --sampling adaptive \
  --h 40 --contact-h 20 --contact-end-h 2 --corner-h 2 --quadrature 12 \
  --dt 8e-6 --duration .045 --backend jax --max-memory-mb 4096 \
  --cache bem/outputs/parsimonious-cache --name parsimonious-corners-dt8
```

原 `/Volumes/Gauss-T7/tatva` 持續只讀。新比較的結果應另存，保留前一輪
[refinement.md](refinement.md) 的歷史資料，不把舊測試改寫成新方法的成績。
