# FEM 能量帳與穩定性驗證範圍

2026-10-25。原h2／float64 FEM重跑已通過完整場核對。
本診斷讀取它的9000筆shear scalar history，計算

```text
Es = 0.5 Ks max(a - p, 0)^2
Wa = integral Ks max(a - p, 0) da
Wn = integral Fn dxn
R  = Wa + Wn - delta(Ebulk + Ekinetic + Einterface + Es).
```

Ks是整個platen的彈簧常數25×200=5000；a為actuator，p為platen位移，
xn為source儲存的normal loading coordinate。依source定義，elastic_energy是bulk能量，
不含platen spring；interface_energy已含法向與切向penalty儲能。
extension_elastic_energy是bulk的子集，不另加一次。

|項目|MPa mm²，單位厚度|
|---|---:|
|Actuator外力功|2832.464816|
|Normal外力功|218.779564|
|Bulk彈性能變化|171.339355|
|動能變化|14.136922|
|Interface儲能變化|0.145020|
|Platen spring儲能變化|−107.670097|
|R：摩擦及數值餘額|2973.293179|

完整欄位、儲存精度與來源指紋見[能量帳紀錄](energy_budget_results.json)。

R包含摩擦耗散及數值／積分誤差，不能當成直接量測的摩擦功或穩定性證明。
保留首末sample，將輸出間距放大2、4、8倍後，R最大改變約0.00615。
這只量測保存資料的work quadrature敏感度，不能取代FEM實際dt收斂研究。
Spring traction重建差由history儲存dtype的roundoff bound檢查，
另外核對HDF5尺寸／mtime仍與原場驗收相同。

目前64接觸子步BEM記錄的micro friction work約2639.075349；
與R的差約334.218，尚不能完全歸因於摩擦功差。
BEM缺少完整bulk儲能／動能與normal-face位移的獨立重建，
所以此處不宣稱BEM總能量帳已閉合，也不由小Newton residual推論完整穩定性。

```sh
PYTHONPATH=bem/src python -B bem/tools/fem_energy_budget.py \
  bem/outputs/fem-event16-authored-f64-h2-4c-interface \
  --field-comparison bem/outputs/fem-event16-authored-f64-field-comparison/summary.json \
  --output NEW_ENERGY_BUDGET
```

輸出須為新資料夾。工具的獨立測試包含固定面的spring充能、
恆力粒子的動能，以及固定spring extension的Coulomb slider，
用來檢查控制座標、功與儲能的符號，避免漏算spring或將儲能當成耗散。
