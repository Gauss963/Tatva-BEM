# 從相同 FEM checkpoint 檢查局部時間步長誤差

完整 graded FEM 在破裂後出現很大的端點累積滑移，尚不能視為動態收斂真值。
`tools/fem_timestep_window.py` 新增一個局部控制：共用原 archived `933ae1f` 的
完整 12-array checkpoint、相同 h2／end 0.015625 mm 網格、材料、penalty、
摩擦與載入政策，在一段固定物理時間窗比較 dt、dt/2、dt/4。
它不是從頭計算的時間收斂試驗，不能據此證明完整 45 ms 已收斂。

原 `run_simulation_dumped` 的 numerical prefix 以私有 globals 編譯，
只在開始定義輸出 helper 前回傳 locals；原 step 與 `lax.scan` 的 AST 不改。
Source SHA、prefix boundary 與剩餘 prefix AST 均檢查，原 module bindings 不改。
模型只建一次；較小步長的 closure 讀取相同 model、另給局部 dt。
完整原載入計畫的物理 endpoint samples 在小步長之間作 P1 插值，
與原步重合的 samples 明確保留原 bit patterns；沒有擬合載入或時間平移。
原 source 的 loading-stop 判斷仍由每個分支各自的接受狀態決定。

換 dt 不能直接沿用 leapfrog 的半步速度。先共同執行一次原步，取得其實際 kick：
`a = (v_half_after - v_half_before)/dt`，然後令
`v_half_new = v_half_old + (dt_new-dt_old)*a/2`。
這保持相同的離散 integer-time velocity，改變未來 half-step 的時鐘；
不用額外呼叫 return mapping，也不改動位移或摩擦記憶。
這是從共用離散狀態出發的 conditional window，不消除更早的時間誤差。

2026-10-30 已完成實際 6000-original-step pilot：
28.623182397–28.634657884 ms，共 11.475488 µs。
原步長 1.912581260414 ns，另兩支為 0.956290630207 ns、0.478145315103 ns。

- 原 dt 分支在兩筆原接觸保存時刻，八個真正 float64 endpoint fields 全部 bitwise 相同。
  原 scalar history 的 16 個非時鐘欄位，轉成原 float32 storage 後也全部 bitwise 相同。
- 每個分支保留所有 fine-step 的 17 欄 float64 scalar history，完整狀態每步都演進。
  接觸場只保留原共用保存端點；此 pilot 的 contact P1 比較區間只有
  28.628181884–28.633183284 ms，不能冒稱全窗的密集場歷史。
- 相對 dt/4，dt/2 的 pressure／signed shear／cumulative-slip 時空 P1 L2 差分別為
  0.000662%／0.001773%／0.012536%；其餘三個完整接觸場也通過局部 1% diagnostic。
- 每步 max-cumulative-slip 的完整時間波形，dt/2 對 dt/4 相對 L2 差 0.016655%；
  從第一筆共同時間起算的增量差 0.045422%。mean-pressure 的微小增量差仍有 2.61%，
  所以絕對場很接近不能取代對動態增量的檢查。
- 0.1 Dc 首次通過的原 dt bracket 為 28.633678643–28.633680555 ms；
  dt/4 為 28.633671471–28.633671949 ms。這是 global maximum 的 threshold，
  沒有據此推斷 native 成核位置。Dc 在這個短窗尚未到達。

8 項本機測試通過，包括 constant-acceleration 的半步轉換、物理載入端點與原 bits、
非法 dt 拒絕，以及禁止用下採樣資料冒充所有 fine scalar steps。
實際三支計算連準備與來源檢查共 233.79 s；外層 serialized control 235.01 s。
驗證時暫停身分已核對的自有 BEM 與 quota controller，完成後恢復；等待仍計入
BEM wall cost。這不是隔離的 FEM／BEM 速度基準。
[完整結果、誤差、時鐘與來源指紋](fem_timestep_window_results.json)保留機器可讀證據。

```sh
PYTHONPATH=bem/src JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 \
  python -B bem/tools/fem_timestep_window.py FEM_RUN CHECKPOINT_NPZ \
  --output FRESH_WINDOW_FOLDER --original-steps 6000
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 \
  python -B bem/tools/compare_fem_timestep_window.py FRESH_WINDOW_FOLDER \
  --output FRESH_COMPARISON_JSON
```

延伸至 29.063076 ms 的控制已完成，但 [破裂後場誤差未通過](fem_postrupture_timestep.md)：
dt/2 對 dt/4 的 pressure／signed shear 時空差仍為 17.68%／10.05%。
上面的短窗結果沒有延伸成整段事件的收斂證書。原 BEM 已恢復。
背景／端點空間收斂、完整原事件時間收斂、BEM 全接觸場與 initialized AD、
以及相同精度下的總成本優勢仍各自保留。
