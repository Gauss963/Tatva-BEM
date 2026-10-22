# JAX 自動微分的現況與設計

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
4. **長時間 production 反向**：45 ms 前向已用完整歷史 FFT；它的 Python／NumPy
   線上狀態尚未接入 AD。完整事件的反向還沒有完成，也未宣稱可用來反演原事件。

## 高效利用的下一步

先保留固定 bulk 預計算，讓摩擦參數掃描重用同一 CQ 權重；這次冷啟動成本主要在
頻率核心，對它們反覆 AD 不會加速同材料／同幾何的摩擦校準。

production 反向需對 dyadic FFT 的線性卷積實作可檢查的轉置，再串接各步隱式接觸
adjoint，按區塊 checkpoint／重算狀態。dense BEM 沒有 FEM 那種局部稀疏性，不能
直接照搬稀疏著色；大系統應使用 `Jv = W0 v + 接觸局部項` 的 matrix-free JVP／VJP，
搭配固定 bulk 預條件器，避免每次完整 dense Jacobian 分解。

如果要反演材料，對 `W=V^-1 H` 使用 `dW=V^-1(dH-dV W)`，重用頻率 LU，
再對 Bessel kernel 提供解析 JVP 或可微分實作；仍需以材料擾動與獨立波速解核對。
等前向的時間／空間精度及切換敏感度建立後，再把梯度用於原事件參數識別。
