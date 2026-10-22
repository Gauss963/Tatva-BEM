# 接觸座標的完整動態縮減

這個選項在 Laplace 頻率域先精確消去線性邊界未知量，再組裝 CQ 權重。
它適用於已建立 static lift、外部法向壓力保持不變的 event-16 擾動。
外部牽引隨時間改變時，需要額外的載入 transfer，目前會明確拒絕。

## 方程

`P` 消去固定支承並約束剛性 platen，完整縮減邊界座標為 `z`。
以 `W(s)=V(s)^-1 H(s)` 定義

```text
K(s) = P.T diag(panel_length) W(s) P
q    = B z = [normal relative displacement, tangential relative displacement, platen]
Z(s) = [B K(s)^-1 B.T]^-1
```

消去其他邊界座標後，擾動的廣義 bulk force 是 `Z(s) q`。
接觸力與彈簧直接作用在 q 上，其功與完整座標中的功相同。
兩個 block 的 compliance 分別求解後相加，避免對沒有耦合的 bulk 子區域做聯合分解。
原有接觸 residual、LSW 更新、Newton、JAX 局部 AD 與隱式根微分均重用。
初始廣義力由 static lift 保存的 bulk traction 取得；改變接觸參數不會默默
重算初始應力。固定初始場的動態律梯度與重新求靜態平衡的參數研究仍是不同問題。

對 `Z(delta(zeta)/dt)` 取 CQ 權重，消去邊界所產生的反射與記憶仍在。
只對當步 `W[0]` 做靜態 Schur complement，無法得到相同的完整歷史方程。
CQ 的 Laplace transfer 離散化依據見
[Lubich, Convolution quadrature and discretized operational calculus I](https://doi.org/10.1007/BF01398686)。
這裡的座標消去式由上述線性方程直接導出；有限 FFT 的 alias 誤差另由對照測試檢查。

## 正確性檢查

- 三個正實／複數頻率：從 q 重建邊界 trace，檢查相對位移、支承、完整 force residual 與廣義功。
- 同 static lift、法向壓力、加速 actuator 的非線性滑動：完整／縮減 CQ 的壓力、
  剪力、塑性滑移、累積滑移與 platen 歷史逐步核對，涵蓋 NumPy 與 JAX 後端。
- 自訂接觸切線及隱式梯度的既有測試繼續保留；另有短程 full/interface 軌跡梯度比較工具。

## 計算成本與輸出

每個 contact panel 只留下兩個相對位移，加一個 platen 座標。
20 mm 案例由兩個 142／144 boundary DOF 歷史變為一個 49 DOF 歷史。
45 ms／8 us 的實數權重約從 1.84 GB 減至 108 MB，未截短時間歷史。
CQ 建立時另使用半頻譜與分列 real IFFT，避免完整 complex IFFT 的大暫存。
資源預估仍保守包含線上 FFT；縮減 preflight 另計未縮減空間核心的幾何與 worker 工作區。

頻率核心仍需完整空間組裝與分解，因此這不是 oblivious CQ 或空間階層壓縮。
幾何、角向 tensor 與完全相同的浮點半徑會重用；Bessel 半徑沒有做量化或近似。
直接徑向 traction 已與原 Green 空間梯度 tensor 的計算獨立核對。
權重快取分開核對頻率 transfer 與 bulk 依賴，純接觸／Newton 修改不會重建相同波傳核心。

縮減 run 保存接觸相對位移、traction、滑移、platen 與載入歷史。
它不保存完整外邊界位移／traction；NPZ 不會放入假的零場或 object/None 陣列。
頻率域 reconstruction 可供檢查，完整時間域外邊界／內部場重建仍待實作。

## 端點加密

`--contact-h` 分別指定接觸面板最大長度；`--contact-end-h` 在兩端以對稱幾何級數加密。
兩個 block 共用分割座標，保持面板配對與相同積分權重。
短 target 面板鄰接長 source 面板時，預先幾何細分 source 積分區間。
近接積分已與較高 Gauss order 對照；self logarithmic subtraction 與 jump 項仍採原公式。

```sh
FDBEM_CORES=4 bem/run.sh event16 --formulation interface --h 20 \
  --dt 8e-6 --duration .045 --backend jax --max-memory-mb 2048 \
  --cache bem/outputs/interface-cache --name interface-h20-dt8
FDBEM_CORES=4 bem/run.sh event16 --formulation interface --h 40 \
  --contact-h 20 --contact-end-h 2 --corner-h 2 --quadrature 12 --dt 8e-6 --duration .045 \
  --backend jax --max-memory-mb 4096 \
  --cache bem/outputs/interface-cache --name interface-corners-dt8
```

縮減方式保留既有離散方程，不會自行改善粗網格物理解。
端點加密案例也未解析整條界面的 process zone；仍需量化空間與時間誤差。

只加密接觸端而保留粗外邊界的測試曾超出初始摩擦界限，不能拿來推進事件。
`--corner-h` 同時解析相鄰角落，避免以新增塑性滑移掩蓋初始化不相容。
這裡 12 點 Gauss 的靜態屈服預估已與 20 點對照；動態積分精度仍須另行驗證。
