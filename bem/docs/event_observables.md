# 成核、前緣與停止載入時間的量化比較

2026-10-28 新增 [`compare_event_observables.py`](../tools/compare_event_observables.py)。
它補充六個接觸場的全界面 P1 誤差：相近的載入等待時間不保證點間傳播時間相近，
單一插值 crossing 或 `argmin` 也不足以決定最早的位置。
完整 graded FEM／2 µs Radau 的結果尚未取得，不能宣稱已重現 FEM 或取得效能優勢。

## 比較方式

三個固定門檻為累積滑移嚴格大於 `1e-7 mm`、`0.1 Dc`、`Dc`，
其中 `Dc = 0.0125959 mm`。分開保留原生節點的最早時間與所有可能的最早位置，
另在兩種方法的節點聯集上比較空間 P1 場的 crossing。
所有表格列出跨越前後的存檔時間，不把線性時間插值當成額外解析度。
進入第一個 shear 觀測值前就已跨越的情形保持 censored；
normal handoff 前的值不改標成 shear `t=0`。

固定量測位置為接觸長度的 10%、50%、90%，本模型即 `y = 48, 240, 432 mm`。
比較量測點的到達時間、兩點間的傳播時間、方向與平均 first-passage speed。
若時間區間包含零，則方向／速度無法判定；即使方向已知，這也不是瞬時局部破裂速度，
而且可能涉及多個前緣。最早時間／停止時間預設相對容差 2%、傳播時間預設 5%，
均為本工具的可調診斷門檻，並非文獻定理。只有區間內每一組可能值均符合容差才 `pass`；
最小可能誤差已超標才 `fail`；其他情形為 `inconclusive`。
不因為兩組存檔資料相同，就把無法解析的短程速度標為通過。

FEM 時鐘由實際 phase step × dt 定義，先核對原存檔時間的四捨五入值。
停止載入使用 FEM 原始 categorical flag 的前後觀測區間；BEM 則核對實際 ramp，
並使用原生權重 coverage 首次達標的 accepted endpoint。沒有擬合時間平移。

預設要求完整、source-frozen、true float64 的 FEM endpoint observations。
另核對兩邊使用同一份已封存的 float64 event-15 plastic-slip／actuator seed、
shear 起始時的 healed memory，以及全界面 `coverage=1` 的停止載入規則；
改成局部區間、速度門檻或其他初態的比較會被拒絕。
量測點另比較真實 endpoint pressure 及有符號 elastic shear 的完整波形，
在時間節點聯集精確積分存檔 P1 曲線的 L2 差；各自扣除第一個共同時刻值的增量差另列，
不能取代絕對場的一致性。零 reference norm 的相對誤差為 `null`。
這些仍不是內部位移／應力／速度波場重建，也不界定未存檔的振盪。

## 已完成的舊版完整事件診斷

[實際結果與檔案指紋](event_observables_results.json) 使用已完成的原 seed、
6 µs／Q12 Radau BEM，對照已完成的 h2 FEM；計算均為 float64，
但這份舊 FEM 的歷程場存為 float32。明確的 `--legacy-diagnostic` 模式以相鄰 float32
數值的中點形成 storage-rounding cells，可能擴大 crossing 區間。
舊座標的 float32 空間誤差未形成界，因此這份結果只能診斷舊版，不能驗收新的 float64 目標。
舊檔案沒有真正的 endpoint pressure／signed shear 觀測值，工具不從 strength／mu 推造它們。

| 原生節點最早跨越 | FEM 區間 [ms] | BEM 區間 [ms] | BEM − FEM 區間 [ms] |
|---|---:|---:|---:|
| `1e-7 mm` | 25.727862–25.732859 | 26.382–26.388 | +0.649141–+0.660138 |
| `0.1 Dc` | 27.248027–27.253024 | 28.782–28.788 | +1.528976–+1.539973 |
| `Dc` | 29.518279–29.523276 | 28.806–28.812 | −0.717276–−0.706279 |

停止載入：FEM 為 `29.783304–29.788314 ms`，BEM accepted endpoint 為 `29.112 ms`；
BEM 早 `0.671304–0.676314 ms`，超過本工具 2% 的時間診斷門檻。
在 `Dc` 門檻，`y=48→432 mm` 的 first-passage speed，
FEM 為 `1919.89–2020.87 m/s`，BEM 為 `1600.00–1684.21 m/s`。
點間傳播時間的最小相對差已達 13.99%，超過 5% 的診斷門檻。
此處的 FEM 仍未做新的 graded 空間收斂，不能把它當作連續解真值。

這些結果說明舊版的差異同時涉及起始時間、後續傳播與停止載入，
無法用一個共同時間平移消除。新的 graded／2 µs 完整比較會使用另一組結果，
不覆寫這份舊版診斷，也不以它宣稱穩定性或速度改善。

## 重現與驗證

從 repository root 執行；`FEM_RUN` 必須是完整的 true float64 observer 輸出：

```sh
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -B bem/tools/compare_event_observables.py BEM_RUN FEM_RUN \
  --output bem/outputs/new-event-observables
```

舊版診斷需額外指定 `--legacy-diagnostic`。輸出必須是新目錄，原 Tatva 保持只讀；
來源、metadata、完整 trajectory／HDF5 以串流 SHA256 核對前後未變。
每個門檻另輸出逐位置 arrival-bracket CSV。

24 項新測試涵蓋 rounding、strict crossing、censoring、最早位置 ties、
時間區間的三態判定、載入等待時間掩蓋傳播差異、方向不確定、波形增量與絕對差分開，
以及 float64 reader 拒絕 float32 shear／排除 handoff 前狀態、初態與載入政策核對。
連同既有 float64 與 earliest-location 控制共 35 項通過。
本機三次序列驗證合計讓同一 FEM 程序等待 6.658164 秒，已記入控制紀錄，
仍包含在 FEM wall time，不另加一次成本。

同日另重新核對並獨立保留了[實際 graded checkpoint 重播證據](graded_prefix_replay_results.json)：
原 2.515059658 ms checkpoint 的 12 個 carry arrays 全部逐位元相同，
複本 HDF5 的壓縮時間資料與 float64 接觸場均實際讀取通過。
FEM 已從該 checkpoint 接續；這只是前段狀態一致性，不是完整事件或 FEM/BEM 一致性。
