# 完整歷史的摩擦子步收斂檢查

`compare_refined_primal.py` 新增明確的 `--mode constitutive-refinement`。
它比較相同完整 45 ms kernel、bulk 時間步、接觸網格、材料、載入、
原始 restart、初始化方法與 source 指紋，唯一允許的數值改變是局部摩擦子步數。
第一個參數必須是子步較多的 reference；原本預設的 solver equivalence 檢查維持原門檻。
另核對原 run manifests 的全部 invocation arguments 與來源指紋，
包含 kinetic root seed；除輸出位置與子步數外，其餘設定必須相同。

```sh
PYTHONPATH=bem/src python -B bem/tools/compare_refined_primal.py \
  FINER_SUBSTEP_RUN COARSER_SUBSTEP_RUN --mode constitutive-refinement \
  --relative-tolerance .01 --output FRESH_COMPARISON_FOLDER
```

工具核對相同的完整整數時間網格、接觸節點、正向映射與初始 bulk trace／force，
比較全部保存的 pressure、signed shear、塑性 offset、累積滑移、μ 與摩擦強度。
P1 時空積分使用整個接觸面及全部時間步，另列最後時間的空間 P1 L2 誤差。
同時比較最大累積滑移、platen、actuator 增量及 spring extension 的時間歷史。
actuator 先扣掉各自的初始值，避免約 10.8 mm 的既有 preload 稀釋載入增量的差異。
這些 scalar 指標會報告，但不取代整個界面場的檢查。

預先設定的 **1% 中間篩選**要求 pressure、signed shear 與累積滑移的
完整時空和最後時間相對 P1 L2 誤差全部通過。若 reference 全零，只有完全零差可通過。
未通過時先保存全部指標，再以 nonzero exit code 結束；不放寬門檻繼續後續 expensive AD。
這個篩選只隔離摩擦積分誤差，不能證明 bulk 時間收斂、空間收斂或 FEM 優勢。

回歸測試包含「最後場完全相同、內部時間有明顯誤差」的反例，
並驗證不同 kernel、時間步、初始 trace、網格與 float32 場會被拒絕。
另對既有完整 45 ms／22501-step 軌跡執行
[相同輸入的 metric identity 控制](constitutive_metric_control_results.json)：
六個接觸場及四個 scalar 的最大差全部為零。
這個控制只驗證工具可讀取完整真實軌跡，沒有假裝成不同子步數的收斂結果。

目前新的 graded N64／N160 完整事件仍在建核；其實際子步收斂尚未建立。
待通過本項及真正 float64 FEM 壓力／剪力比較後，才對兩個完整 primal 執行
包含 static preload 的 reverse VJP／batched JVP 核對，分開量測梯度對子步數的敏感性。
這些 AD 固定 bulk、原始 seed 與記錄的 actuator，對所選 static／transient branches 微分；
不會將此範圍標成材料、幾何或破裂停止時間的完整 AD。
