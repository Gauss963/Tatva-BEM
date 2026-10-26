# JAX 自動微分的現況與設計

2026-10-26新增[完整反射有限bar的custom VJP](radiation_bar.md)：
80,000個細步保留所有返回波，以局部VJP與反向delay ring取代完整ring tape。
與獨立正向JVP最大絕對差1.64e−14。更完整的同程序鏡像對照發現，普通JAX
移除整個history carry的checkpoint後，暫存即由約5.13GB降至16.77MB；
自訂reverse ring再降至8.38MB，這次耗時比不做macro checkpoint的普通AD降低約19%。
因此已修正預設，不將大部分收益誤歸因於自訂VJP。這是1D控制，不是2D event-16效能結論。
同一控制也顯示：平均歷史的末滑移接近、AD內部核對通過，mu_s gradient卻與細歷史反號。
因此完整原算子、時間解析度和gradient收斂須分開驗收。

2026-10-24新增[接觸子步的完整條件式adjoint](substepped_contact.md)：
局部JAX scan配合平衡根隱式微分，反向另傳前一步位移的cotangent。
20步獨立原始Z-history／JAX scan已核對五參數、初始z/c與全部actuator導數，
Python3.11／3.12 CI通過。四／十六子步已完成完整45ms forward；
十六子步的22500步五參數reverse／獨立forward JVP核對現已通過，最大絕對差4.88e-12。
目前最佳64子步完整軌跡也已通過同一五參數核對，最大絕對差7.40e−12；
反向含JIT189.39s，獨立正向核對另外276.73s。
其他cotangent、材料／幾何／lift與event-time的完整核對仍未完成。
最新2 us／Q12原律完整adjoint已算出，但沒有獨立參考，因此不列為本組驗證通過。

2026-10-25新增`resolvent-adjoint --forward-parameter-check`：
以獨立的正向JAX JVP歷史核對五個參數的反向adjoint，保留滑動前kn／kt來源，
以及子步對前一位移的tangent。五個方向共用kernel FFT；
只取frequency-product在最後prefix source以後的輸出，避免future-input circular wrap。
新原型以非對稱隨機歷史直接卷積、原一步／八子步的獨立原始Z scan核對，
Python3.11／3.12 CI通過。2us與4us／N16的完整45ms五參數核對現已通過；
這個選項不核對初始狀態及actuator cotangent，也不擴大材料／幾何／事件時間AD的範圍。
但跨時間網格的gradient差異很大，部分符號相反；每一組內的微分一致性不能
證明物理敏感度已收斂，詳見[時間網格與gradient比較](time_grid_comparison.md)。
固定2us而只增加16到64子步時，部分參數gradient仍變號；
局部狀態／分支解析度的敏感性也必須保留，不能只看AD內部核對通過。

Tatva 原 FEM 已使用 `jit(value_and_grad(elastic_energy_total))` 從能量計算內力，
靜態初始化用稀疏著色的能量 Hessian。這保留了材料模型由能量導出的優勢。
但「內力可微分」不等於 HDF5 runner 已能對完整事件直接求參數梯度；
目前 Python 控制、輸出、載入停止事件和長時間反向記憶體仍需處理。

新增 BEM 現在提供可選的 `--backend jax`。NumPy 解析切線後端保留為獨立交叉檢查。

## 已實作且已檢查

- 局部接觸律是 pure JAX function。`vmap(jacfwd(local_law))` 對每個面板的兩個
  相對位移求切線，再與固定線性體積 operator 組裝，避免對整個 dense bulk 乘法重複 AD。
- Newton 與 L2 line search 在 JIT 內執行，使用 float64；未收斂結果明確回報。
- `lax.custom_root` 對收斂根做隱式微分，不反傳每個 Newton／line-search trial。
- 有界的 `differentiable_rollout` 使用 `lax.scan` 與 `checkpoint`，反傳塑性滑移、
  累積滑移及全部歷史位移的依賴；目前限制 256 步，使用直接卷積供梯度驗證。
- 可以傳入符合 `contact_point` 輸入／輸出介面的自訂 pure JAX 接觸函式，
  切線會自動更新。狀態仍限於塑性滑移與累積滑移；RSF ageing 等額外狀態需要擴充 adapter。
  自訂律的 caller 也必須提供與它一致的初始場；現有 CLI 的靜態 lift 仍使用預設 LSW。
- 開啟／黏著／正負滑移分支與解析切線比對；單步隱式梯度、四步狀態傳遞及
  32 步真實 CQ 核心的五參數梯度，均以中央差分核對。

若平衡式為 `R(z, theta)=0`，標量 loss 的梯度由

```text
R_z.T lambda = L_z
dL/dtheta = L_theta - lambda.T R_theta
```

求得。時間軌跡還需把舊狀態、過去位移和每步 forcing 的依賴串回去；
只對最後一步套上公式，不能算整場模擬的梯度。
這採用 JAX 官方 [custom_root](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_root.html)
的隱式函數微分語意；使用前必須檢查根已收斂。

## 實測範圍

本機 4 worker／thread 設定、JAX 0.9.2、float64 的 32 步固定 bulk 診斷：

|項目|量測|
|---|---:|
|NumPy 單步 root，100 次平均|約 0.473 ms|
|JAX 暖機單步 root，100 次平均|約 0.332 ms|
|JAX 第一次 root，包含編譯|約 0.418 s|
|32 步 loss 加五參數梯度，暖機平均|約 8.70 ms|
|第一次 loss 加梯度，包含編譯|約 0.657 s|
|梯度對中央差分的最大相對差|約 1.8e-6|

數值是小型診斷的一次實測，完整原事件的速度與精度不能由此推算。
`benchmark_ad.py` 同步等待結果，分開首次編譯與暖機成本，遵循
[JAX 計時指引](https://docs.jax.dev/en/latest/benchmarking.html)。
完整數字與版本存於 `outputs/ad-benchmark/summary.json`，並納入事件比較 JSON。

## 尚未涵蓋的導數

1. **材料與幾何 kernel 梯度**：Laplace 核心、Bessel 函數與 CQ 權重目前由 SciPy
   離線組裝。`E, nu, rho`、幾何和 dt 的變動未穿過這一層；不能把現有摩擦梯度當成其梯度。
2. **靜態 lift 對參數的依賴**：目前 rollout 把初始場當固定輸入。
   校準會改變初始化的參數時，必須連同靜態平衡做隱式微分。
3. **切換與事件時間**：開閉、yield、`c=Dc` 和全界面停止載入都不光滑。
   現有梯度是選定分支上的局部導數；跨越切換點的差分不保證一致。
   不能為了讓 AD 好算而默默平滑或更改原摩擦律。
4. **完整事件策略的反向**：新增的 reverse-CQ adjoint 已可對 45 ms、固定記錄載入歷程
   求局部梯度。自動停止載入的事件時間、初始場與材料／幾何的導數仍未包含。

## 高效利用的下一步

先保留固定 bulk 預計算，讓摩擦參數掃描重用同一 CQ 權重；這次冷啟動成本主要在
頻率核心，對它們反覆 AD 不會加速同材料／同幾何的摩擦校準。

reverse-CQ 已對 dyadic FFT 做完整轉置卷積並串接每步隱式接觸 adjoint，
區塊 checkpoint／重算狀態仍待實作。dense BEM 沒有 FEM 那種局部稀疏性，不能
直接照搬稀疏著色；大系統應使用 `Jv = W0 v + 接觸局部項` 的 matrix-free JVP／VJP，
搭配固定 bulk 預條件器，避免每次完整 dense Jacobian 分解。

如果要反演材料，對 `W=V^-1 H` 使用 `dW=V^-1(dH-dV W)`，重用頻率 LU，
再對 Bessel kernel 提供解析 JVP 或可微分實作；仍需以材料擾動與獨立波速解核對。
等前向的時間／空間精度及切換敏感度建立後，再把梯度用於原事件參數識別。

## 2026-10-23：完整歷史的條件式離散 adjoint

`adjoint.py` 使用 JAX VJP 對接觸 residual 和狀態更新求局部導數，
把未來步的 bulk contribution 寫成反向 CQ 權重轉置卷積。
這保留了整段位移歷史的敏感度，不需要儲存或反傳 Newton／line-search trials。
`differentiable_rollout` 的 256 步上限只限制驗證用直接 scan，並不限制這個 adjoint。

對 `R_n(q_n, state_old, theta)=0`、`state_n=g_n(...)`，
當目標為末平均累積滑移，從最後一步往回計算

```text
a_n = g_q.T eta_n - sum_{m>n} W[m-n].T lambda_m
R_q.T lambda_n = a_n
eta_(n-1) = g_old.T eta_n - R_old.T lambda_n
dJ/dtheta += g_theta.T eta_n - R_theta.T lambda_n
```

末 `eta` 的累積滑移分量為面板長度的正規化權重。
residual 的 traction 縮放也轉換到 cotangent 中，另以 actuator 單步擾動差分檢查權重與符號。

角落加密案例共 5625 步、45 ms，反向計算包含 JIT 的本機量測約 2–4 秒。
目標是 **末面板加權平均累積滑移 0.380606 mm**，不是最大滑移。
bulk、static lift 以及該 run 記錄的 actuator 時序保持固定。

|參數|條件式 dJ/dtheta|差分步長|與中央差分相對差|
|---|---:|---:|---:|
|mu_s|48.2713145|1e-7|3.6e-8|
|mu_k|-7.34349946|1e-7|1.9e-6|
|Dc|19.8917745|1e-8|9.6e-6|

較大的 mu_s 差分步長 1e-6 會改變 5／7 個滑動時空分支，差異增加到 2.55%。
這個目標對成核很敏感，局部梯度不保證較大參數步長仍可靠。
此案例的前向解尚未通過空間／時間收斂，因此上述數值僅驗證離散求導，
不能當作已驗證的物理參數識別結果。

```sh
FDBEM_CORES=4 bem/run.sh adjoint bem/outputs/interface-corners-dt8-final \
  --finite-difference --fd-scale .1
```

檢查器會核對全部數值程式指紋、原 case、restart snapshot 及保存的 mesh，
並拒絕不完整或開發中變更 source 的前向結果。詳細向量與每步 actuator sensitivity
保存在 run 的 `adjoint.npz`，數值核對在 `adjoint.json`。

## Galerkin nodal-force backend

`run.sh adjoint` 也辨識新的 Galerkin run，使用對應的靜態 lift、nodal
接觸權重與完整 nodal-force CQ，不把節點誤當面板中點。
短程非線性診斷的 NumPy／JAX 軌跡、scan AD、reverse-CQ 及中央差分已核對。
目標是末接觸加權平均累積滑移；端點權重與壓板共用自由度一併納入。
這項改動保留上述固定 bulk、初始場、載入和分支條件，沒有補上事件時間導數。

## 2026-10-24：JAX-BEM 文獻與材料導數的後續路線

補讀 [Hipperson 等人的 JAX-BEM（2026）](https://arxiv.org/html/2604.21431v1)，
其 §3.2 使用 custom VJP 求平衡根導數，§5 指出完整 BEM 重算與 AD 記憶體仍昂貴。
該工作的聲學形狀最佳化結果不能當作本案摩擦動力學的精度或效能證據。
目前的局部 JAX VJP、隱式接觸根與完整 reverse-CQ 因此繼續保留；
不把所有頻率與所有 Newton iterations 放進同一反向 tape。

對未來的 bulk 材料 AD，可先利用本模型的獨立尺度關係。
固定幾何、Poisson ratio 與積分規則，**不包含 contact penalties 和 platen spring** 的
力／位移 impedance 滿足

```text
Z(s; alpha*E, beta*rho) = alpha * Z(s*sqrt(beta/alpha); E, rho)
E * partial_E Z = Z - (s/2) * partial_s Z
rho * partial_rho Z = (s/2) * partial_s Z.
```

這由 Navier 方程的剛度與慣性尺度直接導出。
新增測試以三組獨立 E／rho 變動，重新組裝兩有限體的靜態 boundary stiffness，
並在兩個右半平面複數頻率核對第一個式子，而非只代入同一函式的導數。
靜態時不能直接反轉 free-body compliance 的剛體零模態；
static lift 的約束／平衡微分須與動態 impedance 縮減分開處理。
這是尺度關係的數值驗證，**尚未實作材料梯度**；也不涵蓋 nu 的改變。

下一步可針對頻率核提供解析或經核對的 `partial_s Z`，並對
`L = inverse(Z+C+K)` 使用 `dL = -L*(dZ)*L`，重用 forward 的分解。
若改變 kn／kt，C 的導數必須同時包含；static lift 的參數依賴仍需另做隱式微分。
既有頻譜插值的導數不能只靠前向取樣誤差作保證，更不能未經驗證就向原 contour 外插。

複數 pullback 也須注意 convention：[JAX 官方 cookbook](https://docs.jax.dev/en/latest/301/cookbook.html)
使用 bilinear cotangent pairing；文獻中的 Hermitian adjoint 必須先轉換到相符的表示。
後續實作需以獨立實／虛部方向的 JVP/VJP pairing、材料擾動及解析波速檢查，
再串接 CQ、初始場與固定分支的完整軌跡。現階段沒有將此列為已完成的 AD 功能。

另新增 `tools/check_actuator_sensitivity.py`，以單一 actuator pulse 的獨立 forward-CQ
JVP 核對指定載入 cotangent，固定 bulk、lift、記錄載入事件時間及接觸分支。
完整粗 Q8 的 index 1724（6.896 ms）forward 為 2.2166394777e-5，
與原 Z reverse 相差 1.58e-14。此參考點支持原 Z 導數；
直接 L 與近遠場 native L 在該點的原 guard 仍失敗，詳見 [完整比較](mixed_quadrature.md)。
檢查器亦在含滑動與非零歷史的耦合 toy 上核對滑動前、開始與後續三個輸入時間。

```sh
FDBEM_CORES=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 JAX_ENABLE_X64=true PYTHONPATH=bem/src python -B \
  bem/tools/check_actuator_sensitivity.py ORIGINAL_Z_RUN --indices 1724 \
  --max-memory-mb 4096 --output NEW_ACTUATOR_CHECK
```
