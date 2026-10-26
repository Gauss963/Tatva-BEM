# 瞬時接觸矩陣的線性成本解法與 JAX 微分

2026-10-26。`tools/structured_radiation.py` 是二維有限 block 的離散 `Cinf` 研究工具。
它尚未接入完整時間積分，沒有改變 event-16 的12.22%滑移差或冷建核成本。
特別是不能用此矩陣取代完整 `Z(s)`、CQ lag zero 或反射歷史。

動機延續 Ciardo、Romanet（2026）的[瞬時／歷史分解](https://arxiv.org/html/2603.00383v1)，
但本文的有限平面應變、混合 P1/P0 係數來自本專案的
[離散輻射極限推導與檢查](radiation_limit.md)，不是沿用其反平面無限斷層係數。

## 接觸線與端點結構

相同材料、相配的垂直接觸網格上，一個長度 `L` 的接觸面板對單一 block 的主導矩陣為

```text
F = L/24 * [[7,5],[5,7]] tensor rho*diag(cp,cs).
u_m = a+q/2, u_s = a-q/2
u_m.T*F*u_m + u_s.T*F*u_s = 2*a.T*F*a + q.T*F*q/2.
```

因此 jump `q` 的接觸線係數是 `L/48`，只連接相鄰節點。
接觸線內部節點沒有連到其他邊界面，消去 mean trace `a` 與其餘邊界，
只會透過接觸兩端及 rigid platen 改變 jump 矩陣。
水平 platen 的內部常數法向速度給出 `rho*cp*width`；端點耦合另保留。
沿用正式座標順序 `(all normal jumps, all tangent jumps, platen)`：

```text
Cinf = blockdiag(Tn, Tt, rho*cp*width) + E*B*E.T
E selects normal/tangent jumps at both contact ends and the platen.
```

`Tn`、`Tt` 是三對角矩陣；`B` 為5×5矩陣。
這是端點支撐的代數結構，不是任意選五個 SVD mode。
工具要求完整 dense constrained-energy reference，核對幾何、對稱正定與重建誤差；
不符合的矩陣會被拒絕，不能無聲刪掉其餘耦合。
以 reference 的 Cholesky 因子 `L` 檢查
`norm(L^-1*(Cstructured-Creference)*L^-T, 'fro')`，
此 Frobenius norm 亦上界相同縮放下的 spectral norm。
預設拒絕超過 `1e-10` 的結果。

這個推導僅針對相同材料、垂直接觸及水平 platen；
不同材料、不同接觸方向或不相配網格需要重新推導。

## 非對稱摩擦 Jacobian

對單一步的原 previous-state 摩擦律，`dp/ds=0`，
`dp/dn>=0`、`d tau/ds>=0`，但 sliding 的 `d tau/dn` 一般非零。
因此基底 Jacobian 在 normal/tangent 分組下是

```text
A0 = [[An, 0, 0], [diag(d tau/dn), At, 0], [0, 0, a*Cplaten+spring]]
J  = A0 + E*(a*B)*E.T.
```

point tangent 必須已乘接觸線積分權重；方程使用 nodal force，不預先按 residual weights 除列。
先解 normal 三對角，再解 tangent 三對角，最後做5×5修正。
轉置解順序相反，不能因為 bulk 對稱就把整個摩擦 Jacobian 當成對稱矩陣。
修正式不求 `inverse(B)`，允許 border 奇異或不定。
每次 solve 的運算／陣列量隨接觸節點數線性增加。

限制是針對**單個 previous-state update**；現有 macro constitutive substeps 的合成切線
可能包含軟化負切線，不能直接假定滿足這些條件。
NumPy API 遇到不支援的切線會報錯；JAX tracing 內遇到不符合條件則返回 NaN，
呼叫端應先使用 `validate_step` 取得明確的錯誤訊息。

137接觸節點、275 DOFs 的 `Cinf` dense coefficients 共605,000 bytes；
三對角與 border coefficients 為4,576 bytes。
這只比較係數，不是 process RSS、XLA 暫存、整段 AD tape 或約40 GB的時間頻譜資料。

## 自動微分

JAX 使用官方 [`lax.custom_linear_solve`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html)，
明確提供非對稱 transpose solver。對 `J*x=b` 的微分仍是

```text
J*dx = db-dJ*x
J.T*lambda = x_bar
b_bar = lambda, J_bar = -lambda*x.T.
```

matvec 保留完整的三對角與 border 項，因此 radiation scale 的導數不會漏掉端點耦合。
AD 不需要回放三對角消去或小矩陣 factorization 的 tape。
這是固定 `Cinf` 的線性方程微分；不代表已完成材料／幾何、static lift、停止事件或完整反射歷史的導數。
三對角 kernel 使用 [`lax.linalg.tridiagonal_solve`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.tridiagonal_solve.html)，
兩節點情況使用常數大小的 dense solve。

## 驗證範圍

測試重新建立兩 block 的完整主導矩陣並作約束能量消去，包含正式事件的
250+254面板、137接觸節點；沒有用剪裁後的矩陣當作自己的參考。
測試要求重建的 relative energy Frobenius 差小於 `1e-11`，
並獨立從 full-body constrained trace 計算能量。

SciPy、JAX 的正向解、JAX forcing VJP 及顯式轉置解，
對照完整 dense 非對稱解，使用 `rtol=2e-10, atol=1e-15`。
摩擦切線來自原 pure JAX law 的 sticking、正／反向 sliding 與 opening 分支，
不是只測 SPD 或零摩擦矩陣。
另外在小型兩 block 測試 radiation scale、三種非零 point tangent 係數、spring 的
reverse AD、五個 forward JVP、dense AD及中央差分。
這些是線性 probe 的導數，並非完整非線性事件的 parameter sensitivity。

一個仍正定、但含接觸線內部 normal/tangent 額外耦合的反例必須被 reconstruction guard 拒絕。
沒有 local runtime benchmark：h1 FEM 仍在本機執行，避免讓新的重負載計時與它重疊。
後續須先量測此 primitive 的實際成本，再接入保留完整 `R(s)=Z(s)-s*Cinf` 的方法，
以完整事件在相同精度下的**總成本**驗收。

```sh
bem/run.sh test -q -k structured_radiation
```

上式由 repository 根目錄執行；本機完整 FEM 控制結束前不要同時跑數值測試。
