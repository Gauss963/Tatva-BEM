# 2026-11-04 補查：長時間穩定、直接時間核與平行儲存

本次補查針對已完成 45 ms 後出現的[全場差異](recovered_full_event.md)。
以下區分原文結果與本專案的實作判斷；沒有以文獻的穩定性或加速數字替代本例驗證。
既有 RK-CQ、摩擦接觸、快速 gCQ 與 JAX 研究見[主文獻筆記](literature.md)。

## Schneider 等：長時間穩定的工業聲學 BEM

[arXiv:2607.02308v1](https://arxiv.org/html/2607.02308v1)，2026-07-02；
[Computational Mechanics 論文](https://doi.org/10.1007/s00466-026-02769-8)。
閱讀範圍：§1–4、MOT 演算法與穩定性比較。
研究是三維外域聲波的 hypersingular 時空 Galerkin；複雜幾何下，
不同積分方程即使同用 BEM，長時間表現也不同。
其有限歷史來自該聲學時域核的 light-cone support，不能當成二維彈性歷史可截斷的依據。
原文的「超過穩定參考最大值一百倍」是數值發散判準，並非本例的 5% 精度門檻。

**實作判斷：**繼續分開核對原動態算子、全場收斂與能量／功；
不能只憑隱式解較平滑就宣稱穩定，也不能把線性聲學結論移植至滑移弱化接觸。

## Capuani：二維彈性波的直接時間域積分核

[arXiv:2605.00234v1](https://arxiv.org/pdf/2605.00234)，提交於 2026-04-30。
閱讀範圍：§1–3、Eq. (3)–(14) 與圓柱空穴範例。
文中由三維核降維，對時間分段線性的位移／traction 積分，建立二維 implicit marching；
以初始靜止、無限彈性介質中的受壓圓柱空穴作控制。
此案例不包含兩有限 block、非零 healed 預載、非線性摩擦或其參數梯度。

**實作判斷：**直接時間核值得作後續替代建核路徑，但目前尚未實作。
導入前須獨立核對 Laplace／時間核、近場積分、預載相容性與完整轉置歷史，
不能直接沿用目前 RK-CQ 的 checkpoint 或 JVP／VJP。
當前先完成同一離散模型的驗證與時間控制，避免同時更換所有誤差來源。

## Zhou、Chen：mCCSR 平行 elastodynamic TDBEM

[Advances in Engineering Software 219 (2026), 104190](https://doi.org/10.1016/j.advengsoft.2026.104190)。
本次取得出版社摘要與章節摘錄，未取得完整正文，故不評估其完整誤差證明。
研究利用時域算子的區塊稀疏與局部對稱，依積分工作量安排 CPU 工作，
並結合多步卷積與 GMRES；示例包括 box、sphere 與 T-beam。

**實作判斷：**原始時域算子的稀疏性不會自動保留在目前凝縮後的有限體 resolvent。
套用前須證明哪些零值是精確因果零值，哪些只是可控近似，並同步驗證轉置／AD。
目前先改善保持相同權重與解的 FFT 重用，沒有據本文刪除任何歷史項。

## 新的對照工具與排程

新增 `compare_kinetic_bulk_response.py`，在相同實際 Laplace modes 下分開比較
FEM／BEM 的黏著與正向完全滑動控制。滑動切線保留 `dτ/dδn = μk Kn`，
不以對稱黏著 Hessian 代替它；這是原接觸律的切線，不是上述論文的新演算法。
局部 2×2 AD／獨立 NumPy map，以及不同線權重下的直接 nodal-force 方程控制均通過。
控制的[來源與實測資源紀錄](kinetic_bulk_math_controls_results.json)已保存。
完整細網格響應已排程，會待前綴驗證釋出記憶體後啟動；
尚未產生驗收結果，不能先指定全場差異的原因。
這些複數 s 的方程只是 frozen resolvent 控制；尚未定位滑動系統的 poles，
也未證明所有取樣 s 都位於該滑動系統的 causal Laplace 收斂域。
