# Radau 建核的磁碟接續

2026-10-28 的[控制與實際細網格檢查](radau_sampling_checkpoint_results.json)。
這項改進用來保留長時間建核的有效工作；沒有改變彈性、摩擦、時間積分或誤差門檻。
完整 event-16 與 FEM 的吻合、時間／空間收斂及同精度效能優勢仍未證明。

## 保存的內容

[`prepare_raw_radau_resolvent.py`](../tools/prepare_raw_radau_resolvent.py) 的
`--durable-build` 使用獨立的 [checkpoint helper](../tools/radau_sampling_checkpoint.py)。
每一個連續矩陣區塊寫入並 fsync 後，才以 SHA256 加入原子更新的 journal。
記憶體 buffer 尚未提交的矩陣會重算；已提交的 prefix 先驗證，之後不覆寫。

原始兩個 Radau 模態保存在 `work-family0.npy`、`work-family1.npy`；
合成的兩個 CQ family 另存在 `compact-family0.npy`、`compact-family1.npy`。
這避免中斷在轉換階段時，已被覆寫的原始模態無法恢復。
個別模態仍保留複數端點，只有合成後的 CQ family 執行原有的實數端點檢查。

取樣的 journal 也保存目前區間與其餘工作堆疊。接續時使用相同的 Chebyshev 節點、
獨立檢查點、raw-family／tail／output acceptance 及原本的細分／直接求解 fallback。
private sampler copy 只更改初始堆疊、已輸出點數與新增輸出 callback；
將這三處還原後，其他 AST 節點必須與原始 sampler 完全相同。
若停在區間中間，會重算該相同區間，逐位元核對已保存的部分，再接著寫入新頻點。
不會以截短的區間重新擬合後，把不同近似接到原資料上。

接續時亦驗證 case、restart、數值套件 source、NumPy／SciPy、原命令參數、
mesh／CQ plan、NPY shape／dtype／layout、連續區塊及實際保存的 bytes。
constructor lock 防止兩個新建核程序同時寫入同一資料夾。
舊版本僅接受已知的 producer fingerprint，以及明確提供的 prefix receipt；
receipt 是來源與 bytes 的紀錄，不是獨立的物理誤差證明。

若中斷在 IFFT／物理座標轉換，只有在兩個完整 compact family 已驗證後，
才移除本建核的未完成衍生 weight 檔，重新生成所有 lag、有限 DFT wrap 與因果載入。
raw／compact 來源仍保留；已有完成 summary 的 kernel 不允許覆寫。
durable 模式的 preflight、progress、journal 與 summary 都以原子更新發佈。

## 完整小型 BEM 控制

七個本機測試通過，涵蓋：

- 接受的插值區間中途停止，接續後所有頻點逐位元一致。
- 已保存矩陣損毀時拒絕接續，且不改寫資料。
- 真正 BEM 建核在 raw 取樣、compact 轉換、物理 weight 後處理三階段中斷。
- 舊 producer receipt 接續、只讀驗證、參數／套件／runtime 改動拒絕與互斥鎖。

三種中斷與舊版本接續的最終結果，均與不中斷的 producer 逐位元一致：
兩個 compact spectra、兩個 time-weight families、shift wrap、初始阻抗與
`elastic.npz` 中的所有因果載入陣列。原本的獨立原始 BEM mode／stage 與
lag-zero composition 門檻仍是 `1e-6`；沒有放寬。
這些是小型完整建核控制，並非完整細網格破裂或 FEM 一致性證明。

## 實際 event-16 細網格

以只讀的 `--verify-resume-only` 檢查目前保留的 2 µs、45 ms、Q12 細網格：

|檢查項目|結果|
|---|---|
|retained DOFs／FFT size|531／65536|
|slow branch 已提交頻點|256／32769|
|已保存 bytes|1,154,912,256|
|prefix SHA256|`f3802c5c698bed22e55be770bb4672ee9ec86d50c9ab24f1e9f1fc982c905801`|
|來源、runtime、參數與 plan|通過|
|驗證期間改寫建核資料|無|
|durable 建核總磁碟預估|693.24 GiB|
|已占用空間扣除後尚需|692.17 GiB|
|建核陣列 peak 預估|18051.50 MiB|

額外保存原始模態比原本 in-place 模式多用約 275 GiB，磁碟預估已包含這個成本。
目前約有 982 GiB 可用；實際啟動時會再檢查。
陣列預估不等於硬性 RSS 限制；CQ plan 中約 84 GiB 的 dense-history 預估亦不代表
這個 bounded disk builder 會配置該陣列。
建核將在獨立完整 FEM 之後接續，避免超出四核心及本機記憶體預算。

## 使用方式

新建核使用新的資料夾；接續時重複相同參數並加上 `--resume`：

```sh
python -B bem/tools/prepare_raw_radau_resolvent.py "$CASE" \
  --output "$KERNEL" --dt 2e-6 --duration .045 --workers 2 \
  --primitive-partition --weight-layout fortran \
  --max-workspace-mb 10240 --max-memory-mb 18432 --durable-build
```

`--resume --verify-resume-only` 只讀取、驗證與印出資源結果，不開始取樣或建立 journal。
舊的無 journal 資料另需 `--resume-receipt "$RECEIPT"`。
未完成嘗試、來源遷移與 legacy receipt 均保留在 journal。
journal 的計時從 model/resource preflight 後開始；完整成本應採外層 invocation 計時。
缺少結束事件的嘗試不會被視為具有精確總時間。

本機測試與細網格讀取檢查期間，FEM 與其 CPU 控制器依序短暫停止、再恢復同一 PID；
所有這些等待均保留在 FEM wall interval 內，包含第一輪失敗的測試時間。
不把這些控制或只讀驗證當成建核／FEM 的速度優勢。
