# 保留局部 AD 的接觸切線組裝

2026-11-03：可選工具 `tools/identity_stage_tangent.py` 減少接觸切線的組裝成本。
它尚未用於完整細網格軌跡；沒有證明全程 FEM 一致性、速度或穩定性優勢。
[數值、完整粗網格輸出、來源與 CI](identity_stage_tangent_results.json)保存本輪證據。

## 方程與 AD

原方法以 JAX `vmap(jacfwd(local))` 微分每個接觸點的兩段子步律，保留
兩個 stage 之間的法向／剪向及狀態耦合。對目前 normal／shear jump 是 identity
選取的座標，原來的 16 次稠密接觸矩陣乘積可改為直接放入相同位置。
265 個接觸點只有 **4240 個接觸切線 entries**；原完整 1062×1062 bulk 矩陣仍保留。
求根 residual、Newton／line search、摩擦律、子步與接受門檻均不變。

工具只在精確符合 identity maps 時使用此分支；一般插值保留原組裝。
scatter indices 的唯一性會實際檢查後才傳給 JAX。非有限局部導數亦退回原方法。
這符合 [JAX Array.at 的使用條件](https://docs.jax.dev/en/latest/_autosummary/jax.Array.at.html)。
實驗在 private binding 中使用原類別，不更動執行中的既有模擬或 package。

這項改進保留 Tatva 的 AD 設計：局部律自動微分，已收斂根用 implicit AD，
完整歷史使用原轉置算子。它沒有增加材料／幾何或 endogenous stop-time 的梯度能力。

## 已完成的控制

六項測試通過：stick、slide、reverse、open 的切線分別與原組裝及
獨立 full-residual `jacfwd` 核對；另驗證耦合平衡根、非零 Dc 梯度與一般 maps fallback。
數值來源 `9c974c7` 的 [完整 CI](https://github.com/Gauss963/Tatva-BEM/actions/runs/41572063502)通過。

相同已保存 kernel、相同短 lag cache 的完整 128 步粗網格控制中：

- 位移、接觸 response、全部 stage 載入梯度、七組 JVP 逐位相同。
- 五參數梯度最大絕對差 `3.31e-23`；原 `rtol=1e-7, atol=1e-18` 門檻通過。
- JVP／VJP 的 mixed normalized errors 為 `4.45e-6`／`4.73e-6`，均小於 1。
- 最大累積滑移為 1.03382 mm，超過該控制的 Dc；沒有重新組裝任何頻率。

粗網格為 h=160、5 個接觸點、dt=2 µs、診斷載入 7812.5 mm/s，並非原 20 mm/s 的事件。
另用實際細網格的初始 lift、bulk 矩陣與 `(64,128)` 子步，在三個代數試算狀態核對
全部切線 entries，與原組裝的差為零；獨立 residual JVP 的相對差不超過 `2.59e-16`。
其中 shear perturbation 的命名不代表已驗證細網格滑動軌跡。

## 計時範圍

每個細網格試算狀態配對重複三次，交換執行順序並等待 JAX 結果就緒。
切線組裝 median CPU 時間由約 0.105–0.107 s 降至 0.00657–0.00784 s，
時間比為 **13.66–16.04**。首次編譯另記：原版／新版 CPU 約 1.47／1.71 s。

這些是在原事件 3.85 加上診斷 0.1 的平均 CPU 配額下量測的**單項 CPU 成本**，
不是隔離四核心的 wall-time 比較，也不能乘上建核加速比推估整場速度。
全程仍需計入初始化、完整 CQ、求根、拒步、編譯與驗證成本，並通過相同精度的 FEM 對照。

原細網格 cached 事件已在 28.902 ms 失敗；該次尚未使用此切線工具。
[全程狀態](full_original_radau_progress.md)保留其失敗與原基準恢復紀錄。
