# 可續跑的 body／probe 頻率核心

2026-10-29，內部觀測 driver 已可讀取預先建好的完整 rectangular 頻譜，
並以原有[有界 CQ／完整轉置](rectangular_cq_history.md)重建位移、速度與應力。
這解除原直接 sampler 的小型頻率數上限，尚未證明完整 FEM 事件相符或更快。
原 Tatva 只讀；目前執行中的 FEM 與原 2 µs 核心排程不修改。

## 求解與取樣

[`interior_raw_sampling.py`](../tools/interior_raw_sampling.py) 每次 raw evaluation 共用
兩個 body 的 V／H／D 及各 probe 的 single／double potential 與 target gradient。
body 接觸／platen trace 固定後，以既有兩級 Schur 消去恢復完整邊界位移，
再解 `V lambda = H u` 得到 Somigliana auxiliary density。
位移為 `Vp lambda - Kp u`；velocity 乘上 Laplace s，gradient 轉為 plane-strain stress。
這避免先反轉 free-body compliance，沒有正則化、頻率切換或人工阻尼。
三個複數頻率的短程測試與原 free-body 代數相符；完整事件的數值穩定性優勢仍待量測。

新的矩陣有 96 output rows、531 fine-mesh inputs，座標為 physical `Tq`。
Chebyshev degree、raw checks、tail、partition、direct fallback 與所有 grid emissions
沿用原 sampler。AST adapter 只更改 rectangular shape、加強 condensed error check，
及恢復原 sampler stack 的 bookkeeping；恢復這些指定修改後須與原 AST 完全相同。
每個接受區間同時檢查原 Frobenius norm 與**每個 output row**，避免應力的尺度
掩蓋位移／速度。原 raw／output 相對容許值仍為 `1e-11`／`1e-9`。
這是經驗性插值檢查，沒有均勻誤差定理。

這項實作借鑑 [Melenk、Nick 的 parsimonious CQ](https://arxiv.org/html/2410.15079v1)
減少 Laplace 組裝的方向，並非該文 multistep 方法與複雜度證明的直接移植。
每個 Radau mode 都獨立取樣，所有需要的頻率仍轉換成 retained field map。
目前不壓縮空間 rank、不刪除反射歷史，也不使用 source trajectory 決定插值。

## 可續跑的儲存與驗證

[`prepare_interior_spectra.py`](../tools/prepare_interior_spectra.py) 保留兩份 raw modes，
另寫兩份 compact F0／F1。兩 mode 依序建立，共用當前 body／probe geometry。
rectangular journal 重用既有連續 chunk commits、durable sampler stack 與
bitwise prefix writer；獨立 kind／shape 不接受舊 square kernel 的 receipt。
每個 chunk 先寫完並 fsync，再發布 journal；resume 核對 source、case、seed、runtime、
headers、完整 committed hashes，重算 active interval 並逐矩陣核對既有 prefix。
compact families 若在不同 prefix 中斷，從兩者較小的 prefix 重算及核對。
不補零、不捏造缺失資料。

建核時先以原 free-body algebra 對獨立 spot frequencies 檢查，再合併成 compact families。
高頻的遠處 probe rows 可指數衰減至近零，因此 audit 明列 global **及** row 的
`rtol=1e-6, atol=1e-12`，另保留帶 `1e-30` floor 的相對差診斷。
不能把 mixed tolerance 的通過解讀為所有近零 row 都達到相同的相對精度。
中斷程序缺少 outer end time 時，完整總耗時標記為 unaccounted；resume 時間不替代原成本。

[`check_interior_spectra.py`](../tools/check_interior_spectra.py) 是另設預算上限的小型稽核：
在**每個** mode／frequency 重建原 free-body map，同時檢查 raw 與 compact maps。
可另外以原 stage symbol 展開、獨立 dense IFFT、完整因果權重重建全部 source stages，
核對 prepared observations。稽核前後核對所有 inputs 的串流 SHA，拒絕執行期間的變更。
完整細事件超出此 direct-audit 預設上限，不會自動降採樣或縮短比較時間。

## 已做控制與限制

[結果、成本與指紋](interior_raw_sampling_results.json) 記錄所有嘗試。
39 項相關測試包括原解析 P／S 波、dense CQ、完整轉置、JIT VJP、逐列／global guard、
接受插值區間的 prefix replay、損毀拒絕，以及實際雙 block fresh／raw 中斷／compact
中斷三條完整路徑。續跑 compact files 的 SHA 與 fresh 相同，靜態 lift 不變。
另加入單一時間步控制，確認 audit indices 不會超出小型頻率網格。

較長控制使用 Q12、h／contact／end／corner 160 mm、11 inputs、12 probes，
2 µs × 257 steps、FFT 1024。慢速 mode 有 869 次 raw evaluations，未接受插值；
快速 mode 有 391 次，384／513 grid points 使用插值。
因此總 raw 次數 1260 **高於**直接取樣的 1026 次，另有獨立 audit 成本。
這個案例沒有證明總建核加速；不能只引用快速分支省下的次數。
兩 mode 的全部 1026 次原始代數檢查，在 `rtol=1e-7, atol=1e-12` 的 global／row
mixed norm 下最大 normalized error 為 `0.00946`（通過門檻為 1）。
全部 257 步增量相對獨立原始 map／dense causal CQ 的 L2 差為 `2.76e-14`；
prepared driver 額外 BEM 頻率求解為零，完整轉置 directional error 為 `4.10e-16`，
JAX fixed-map VJP 與直接轉置差為零。原 f09 seed 與完整 initial residual 均保留。
此來源的最大累積滑移為零，仍屬 sticking 控制。
此時間範圍與網格是數值實作控制，不能替代 fine nonlinear rupture、FEM 或事件傳播比較。

prepared driver 的正常 forward 不重做 BEM 頻率求解；`--check-ad` 額外檢查完整歷史
transpose 與固定核心 JAX VJP。AD 範圍仍是 retained perturbation stages。
原靜態 lift、初始 residual 與 source trajectory 不改；initialized observation gradient、
材料／幾何導數與完整 FEM 波場收斂仍未完成。

## 完整範圍 preflight 與使用

45 ms、1 µs、531 inputs／96 outputs、FFT 131072 的 preflight 只建立 metadata，
沒有執行頻率組裝或建立大型 NPY。四份 raw／compact 頻譜加 reserve 需約 199.378 GiB，
兩 worker 的 array estimate 約 18163.2 MiB，含 256 MiB probe／其他 reserve。
這是邏輯 array estimate，沒有量測完整 process RSS。
亦不能讓此核心、既有大核心與原方程稽核的暫存同時超過本機磁碟／RAM。
完整 1 µs 工作尚未啟動。

在 repository root，對自行選定且通過資源檢查的新目錄：

```sh
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -B bem/tools/prepare_interior_spectra.py \
  bem/src/finiteblock_bem/resources/cases/pmma_event16.toml \
  --output NEW_INTERIOR_KERNEL --dt 2e-6 --duration .045 --workers 2 \
  --max-workspace-mb 10240 --max-memory-mb 18432 --preflight-only
```

`--preflight-only` 的目錄不可轉為 production；實際建核須使用另一個 fresh 目錄並移除該旗標。
對已開始但未完成的建核，保持完全相同參數，加入 `--resume`；只核對 prefix 可再加
`--verify-resume-only`。對相同 mesh／dt／horizon 的完整原 primal：

```sh
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_ENABLE_X64=true \
python -B bem/tools/reconstruct_interior_reference.py COMPLETE_PRIMAL \
  --prepared-kernel COMPLETE_INTERIOR_KERNEL --output NEW_OBSERVATIONS \
  --output-rows 1 --max-workspace-mb 8192
```

完整 source horizon、probes、環境及 kernel source 指紋不符即拒絕。
可選 `--check-transpose`／`--check-ad` 增加核對；一般 forward 不自動做多次 FFT。
達成原目標仍需要完整細網格事件、true-float64 FEM probes、空間／時間／梯度收斂，
以及相同精度下的穩定性或隔離量測的總耗時優勢。
