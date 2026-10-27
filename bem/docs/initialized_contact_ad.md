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
單子步與四子步分開。真實完整 event 的 coarse-P1 Γ8 控制現已通過，見下方；
original-P1 Γ8 仍在序列執行。材料／幾何與載入停止時間導數仍未包含。

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

## 完整離散 AD 通過，但敏感度尚未收斂

2026-10-27 完成固定初始化、coarse-P1 seed 的 Γ4／Γ8 全 45 ms conditional AD。
兩者 reverse／五方向 scalar JVP 最大差分別為 3.16e-11／1.12e-11。
其解的最大累積滑移僅相差約 0.000129 mm，梯度卻有大的變化：

|參數|Γ4 reverse derivative|Γ8 reverse derivative|
|---|---:|---:|
|mu_s|129.65743|52.80832|
|mu_k|−7.06022|−8.95183|
|Dc|33.68019|9.64383|

目標為末端接觸權重平均 cumulative slip，導數的單位為 mm／各參數單位。
每組完整原始數據、版本指紋與逐參數差見
[contact_gradient_resolution_results.json](contact_gradient_resolution_results.json)。
不同接觸點／active set 的條件式導數尚未收斂，不應用於物理參數識別。
正反向一致只能驗證同一離散、同一分支的微分一致性；短程獨立有限差分
雖已通過，完整事件的有限擾動、初始場與 mesh／time 敏感度仍要分開檢查。

Γ8 原布局耗時為 reverse 15818.89 s、五方向 forward 1399.37 s，
完整讀檔／hashing／輸出總計 17226.93 s；peak RSS 9034.50 MiB。
這是包含 IO／JIT 的程序計時，並非硬 CPU 隔離或同精度 FEM 速度對照。
新的 [layout-aware history](layout_history.md) 已對相同 Γ4 primal 核對
全部五參數梯度與 22501 個 actuator cotangents，現正繼續 initialized controls。

## 包含初始化的完整 Γ8 檢查

coarse-P1 Γ8／N64／45 ms 的 initialized reverse／batched-forward 檢查已通過，
五參數梯度最大差 1.684e-11。與先前 conditional 檢查使用完全相同的 primal SHA-256，
但初態導數的定義不同，不能要求兩者 gradient 相同：

|參數|固定初始化|包含 selected static active set|
|---|---:|---:|
|kn|9.21467e-7|6.46159e-8|
|kt|1.22279e-5|1.85402e-5|
|mu_s|52.80832|52.81354|
|mu_k|−8.95183|−8.95183|
|Dc|9.64383|9.64383|

目標與導數單位同前。healed strength 與零 cumulative memory 不依賴 mu_k／Dc，
因此這兩項的初始化貢獻為零；它們數值上保持一致。
kn／kt 則顯著改變，不能把固定初態的 conditional gradient 當成完整輸入導數。

本次 reverse 484.69 s、五方向 batched forward 716.22 s，包含讀檔、static rebuild、
hashing 與 JIT 的總時間 1229.33 s，peak RSS 8570.64 MiB。
[完整結果](initialized_coarse_full_ad_results.json)保存 scope、逐參數對照與來源指紋。
這仍不是材料／幾何、event-time 或跨 active-set 的導數；也沒有解決已觀察到的
mesh／time gradient convergence 問題。original-P1 Γ8 正在後續執行。
