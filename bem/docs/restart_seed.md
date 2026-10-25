# 原始float64 restart的固定網格對照

2026-10-25。既有BEM使用的CSV是由FEM的float32界面輸出匯出；
原event-16模型實際讀取的是event-15 HDF5的float64 `final_state/plastic_slip`與actuator。
這個輸入精度差須量化，不能因數值很小就先假定對rupture沒有影響。
目前尚未執行新的完整事件，也沒有宣稱它會消除既有13.2%的滑移差。

`restart_seed_experiment.py`只在完整原精度FEM重跑完成且來源未變更時，
唯讀擷取原restart的少量陣列。它核對wrapper已保存的seed指紋、HDF5大小／mtime，
以及新FEM第一筆界面塑性滑移在儲存精度下的相等；不複製完整HDF5。
累積滑移依原event-16的完整healing／reset規則設為零，actuator保留原float64值。

`resolvent-simulate --restart-seed`允許明確指定這個輸入對照，但要求：

- 所有陣列為有限float64，y節點與原CSV逐位相同且嚴格遞增。
- 塑性滑移與y同形，累積滑移為零，actuator與原載入逐位相同。
- 原case／package／kernel指紋檢查照常執行；不更換幾何、材料、支承、penalties或時間網格。
- 重新求static lift，另保存實際seed檔案及provenance的SHA256；輸入變更會拒絕接受該run。

bulk transfer只讀取同一mesh／材料／constraints，與restart數值無關。
回歸測試以同一實際雙block transfer直接核對修改前後矩陣逐位相等；
沒有手動改寫舊kernel的來源metadata，也沒有放寬任何cache guard。
這不是更換FEM初始內部場的完整checkpoint；該空間初始化誤差仍要獨立檢查。
adjoint亦拒絕比較不同explicit seed的run。

以下流程排在獨立FEM計時與原檔場核對之後；輸出使用新資料夾：

```sh
PYTHONPATH=bem/src python -B bem/tools/restart_seed_experiment.py \
  bem/outputs/fem-event16-authored-f64-h2-4c-interface \
  --output bem/outputs/event16-original-f64-seed
FDBEM_CORES=4 bem/run.sh resolvent-simulate bem/outputs/direct-resolvent-fine2-q12 \
  --constitutive-substeps 16 --contact-root-recovery --fft-cache-mb 8 \
  --max-memory-mb 8192 --restart-seed bem/outputs/event16-original-f64-seed/restart_seed.npz \
  --output bem/outputs/galerkin-fine2-q12-contact16-original-seed
```

比較時必須標明seed來源，不把不同輸入的差當成純時間／子步加密誤差。
