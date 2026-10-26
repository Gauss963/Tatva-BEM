# 包含 healed 初始化的接觸參數梯度原型

`initialized_refined_adjoint.py` 延伸既有完整 causal AD，引入 static active-set
線性根對五個接觸參數的 Jacobian。bulk、原始 restart 輸入與 recorded actuator
schedule 固定；static／transient 分支已選定，不宣稱跨分支或 event-time 可微。

除了初始位移與 plastic offset，實作保留兩個容易漏掉的來源：

1. 初始位移的改變對 bulk force 是 `Z_static * dq0`，不是動態瞬時 `Z0 * dq0`。
2. 每個後續修正歷史都含 initial contact force 的導數；只傳 current traction 不完整。

用 `z_relative = z_physical - delta_z_initial` 作狀態座標，可讓既有正／反向
因果引擎維持零 initial tangent。局部 response 明確傳回初始 pressure/shear
供修正來源微分，並在 root residual 加入 static 與 instantaneous dynamic
bulk 矩陣之差。Newton 與 active-set 迭代都不反傳。

新增測試從獨立 original-Z 歷史求解：每次參數有限差分都重新解 healed static
平衡，再執行完整短程動態，核對五個接觸參數及 forward/reverse agreement；
單子步與四子步分開。這是新原型，真實完整 event 驗證尚待完成，沒有
切換目前 frozen 的完整事件研究。材料／幾何與載入停止時間導數仍未包含。

static sensitivity 的來源 `7810755` 與含初始化的完整短程來源 `a2f1a77`
已分別通過 Python 3.11／3.12 的
[static CI](https://github.com/Gauss963/Tatva-BEM/actions/runs/40359292659) 與
[initialized-history CI](https://github.com/Gauss963/Tatva-BEM/actions/runs/40360167259)。
這些成功不等於真實 45 ms 梯度收斂。

同一初始化 context 亦可注入 `batched_forward_gradient`，共用五方向的完整
kernel FFT。測試會把它與五個 scalar JVP、reverse VJP 及重新初始化的
original-Z 有限差分分開核對；運算次數下降不代替完整計時。

`check_initialized_contact_ad.py` 可讀取完成且來源未變的 `fem-active-set` primal，
重建並核對 static lift、保存狀態、完整權重與物理輸入，再執行 reverse／batched
forward 核對，結果另存新目錄，不改寫 primal。執行時使用與原模擬相同的四核心
thread environment，並與其他 heavy 求解序列執行。例如 Python 環境已設定後：

```sh
PYTHONPATH=bem/src python -B bem/tools/check_initialized_contact_ad.py \
  COMPLETED_PRIMAL --output NEW_AD_OUTPUT
```
