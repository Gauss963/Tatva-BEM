# 原始 Green 核的內插階數選擇

2026-10-27。新的 graded event 保持 Q12、2 us、完整 45 ms／65536 FFT circle、
所有 lag、物理模型與既有 raw/output tolerances。先前為控制配置而選的 degree8
可能反而增加直接取樣；已依下列 primitive screen 改回 degree16。
這不是已完成的完整 BEM 加速比較。

raw sampler 同時檢查獨立點誤差及最後四個 Chebyshev coefficients。
在 degree8 時，這包含第 5–8 項，故降低 degree 不只降低每段配置，
也會使尾項條件更難通過，增加細分與 direct fallback。
實際評估次數必須量測，不能只以每段的內插節點數推估成本。

`screen_raw_interpolation_degree.py` 在相同 circle 上取實際 PMMA Green scalar
primitives，距離為 0.015625、0.125、1、10、100、500 mm。
emitter 固定為 identity，以隔離 raw check／tail 的行為；它沒有 body assembly、
Schur condensation、非線性摩擦或原方程的誤差界。
相同 raw tolerance 1e-11 下的結果為：

|Degree|Primitive evaluations|Direct grid points|Interpolated grid points|
|---|---:|---:|---:|
|8|24838|12288|20481|
|12|10594|4224|28545|
|16|3742|128|32641|
|24|1687|128|32641|

這是階數選擇的診斷，不能將次數比當成 full solver speed ratio，
也沒有證明 full-matrix amplification 會相同。
獨立的 portable CLI 重跑與初始診斷的全部 count records 完全相同；
單元測試另核對明確指定的 FFT circle 與不宣稱 BEM accuracy 的輸出契約。
[完整數據與指紋](raw_degree_screen_results.json)保留兩者的核對及資源 preflight。

degree16、兩 workers 的候選 fine model，保守陣列估計為 17776.14 MiB，
已通過 18432 MiB budget，interpolation workspace 為 8563.83 MiB。
程序 RSS、OS file cache 與陣列預估分開記錄；本機四核心預算不代表硬 affinity。
新 kernel 使用新的目錄，degree8 的部分資料與 source／log 全部保留。
舊建核 stage 已花 2870.70 s，包含小模型驗證時的 32.46 s 暫停；
後續 campaign 的付費建核成本須加上這次嘗試，不能只量最後成功的一次。

```sh
# 不需要 FEM HDF5；預設 circle 即本次研究使用的參數。
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 python -B \
  bem/tools/screen_raw_interpolation_degree.py --output FRESH_PRIMITIVE_SCREEN

# 或從既有 kernel/preflight metadata 讀取 circle；不讀其大型矩陣檔。
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 python -B \
  bem/tools/screen_raw_interpolation_degree.py --plan SOURCE_METADATA_JSON \
  --degrees 8 12 16 24 --output FRESH_PRIMITIVE_SCREEN
```

完整新建核仍逐點產生所有 FFT matrices，保留原 independent output checks；
完成後仍須做完整事件、原始因果方程稽核與 graded FEM 對照。
