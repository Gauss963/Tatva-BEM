# 直接保存 CQ 插值與磁碟時間權重

2026-10-24。此實驗讓首次建核也能避免完整稠密頻譜，不只在建核之後壓縮快取。
它保留所有頻率格點、完整因果啟動及每個所需修正 lag，目的是支援更細時間步的收斂研究。
目前尚未以此 backend 完成原 45 ms 事件，亦未證明與 FEM 等精度的速度優勢。

## 方法與誤差範圍

沿用原自適應 Chebyshev sampler 的節點、獨立檢查點、tail test、區間細分與直接計算 fallback。
透過私有函式 binding 保存已接受區間的 17 個矩陣值、原 barycentric weights 及精確 normalized coordinates。
直接計算區間保存原矩陣；完整頻率圓不會先配置在 RAM 或磁碟。
此步不增加另一層多項式近似，也不改套件的 module globals。

每一個插值格點都核對磁碟表示的重建值，條件為
`||decoded - original_interpolant||_F <= 1e-18 + 1e-13*||original_interpolant||_F`。
這個檢查的來源是**原 sampler 的插值結果**，不是每個格點獨立重算的 BEM 值。
原有取樣誤差仍是有限檢查點的經驗驗證，沒有連續頻率誤差定理。
標準 CQ 的實軸端點投影、FFT contour 與 damping amplification 維持原式。

另提供逐列 IFFT 到唯讀 NumPy file map。每個所需時間權重都保留，
不配置完整普通 ndarray；可用 `madvise` 的系統會在每批寫入後釋放 mapping pages。
workspace estimate 是陣列估計，不能當成程序 RSS 或 OS file cache 的硬上限。
初始化使用完整因果彈性 convolution；矩陣 prefix 只省略非線性修正開始以前的零 source，
不刪除非零長尾。

檔案與來源以 SHA-256 核對。不完整覆蓋、重複寫入、來源變更或超過容量預算時不寫完成標記；
時間權重以 exclusive creation 建立，拒絕覆寫既有 kernel。

## 實際 BEM 的 512 步診斷

這是雙 block、h=80 mm、Q12、15 retained DOFs、dt=0.5 us、總長 256 us、
actuator 增量 2 mm 的**加速滑動診斷**，不是原 45 ms 載入事件。

|項目|結果|
|---|---:|
|完整半頻譜格點|1025|
|實際 BEM transfer evaluations|848|
|插值／直接表示格點|681／344|
|已接受插值區間|6|
|原稠密陣列資料量|3,690,000 bytes|
|native 陣列檔案資料量（含 NPY headers）|1,623,960 bytes|
|相對原 sampler 插值格點的最大正規化儲存誤差|0|

相對於另跑的 dense-frequency CQ：滑移最大差 1.66e-13 mm、剪力差 2.22e-11 MPa，
五參數梯度最大相對差 2.23e-12；actuator 與兩種初始狀態 cotangent 亦通過檢查。
滑動、開閉、弱化 clip 與等值分支變更皆為零。
1／4-worker 的解析非對稱矩陣測試也核對原取樣 records 完全相同、每個頻率與時間權重一致。
更短 32／64 步 BEM 診斷只有直接區間，不能用來證明插值路徑；512 步案例補上此項覆蓋。

AD 範圍仍是固定 bulk、lift、記錄載入與 active branches 的條件式導數，
不包含材料、幾何、初始化及 coverage hold 時間。
診斷與四-worker fine2 建核同時進行，所記 wall time 不構成受控效能比較。

同一診斷另以**實際 BEM 直接建立 work-resolvent**，頻率取樣不經原 Z spectrum：
800 次 transfer evaluations、5 個插值區間，陣列檔案 1,562,632 bytes。
時間權重逐列寫到磁碟，再獨立完成接觸推進；對原 dense Z 軌跡，
滑移差 1.04e-10 mm、剪力差 1.44e-8 MPa，五參數 adjoint 最大相對差 1.13e-9。
所有精確分支相同，actuator／初始狀態 cotangent 亦通過原容差。
瞬時 L0 inverse composition 的無窮範數誤差為 3.99e-11。
這也驗證 prepared standalone trajectory 可以明確指定原 Z 作為獨立 AD 參考。
原始數字、指紋與驗證範圍保存在 [結果紀錄](native_spectrum_results.json)。

## 使用

從 repository 根目錄執行，輸出必須是新資料夾：

```sh
# ORIGINAL_Z_RUN 須先有 run.sh adjoint 產生的獨立參考。
FDBEM_CORES=4 bem/run.sh native-spectrum-check ORIGINAL_Z_RUN \
  --operator original --workers 4 --native-spectrum NEW_SPECTRUM \
  --output NEW_DIAGNOSTIC

# 由實際 BEM 直接建立新時間網格的 work-resolvent、磁碟時間權重與完整啟動。
FDBEM_CORES=4 bem/run.sh native-resolvent-kernel SOURCE_RUN \
  --dt 1e-6 --duration .045 --correction-duration .020 \
  --workers 4 --max-memory-mb 8192 --max-spectrum-mb 16384 \
  --output NEW_KERNEL
```

第二個命令只準備 kernel，並沒有自動完成時間推進或 FEM 驗證。
`--max-memory-mb` 限制建核陣列估計，`--max-spectrum-mb` 限制 native 頻譜磁碟容量；
另外仍需要足夠空間保存完整時間權重。
另有[指定容量的線上 dyadic FFT cache](bounded_history.md)，已通過完整粗網格的解／AD 等價核對。
fine 1 us 尚未完成建核與事件驗證，不能據此宣稱本機已完成這個解析度。
原 2 us／45 ms 四-worker 計算仍沿用原 backend，兩者不混用。

實作在 `tools/native_spectrum.py`、`tools/prepare_native_resolvent.py`；
相較既有[離線壓縮](polynomial_spectrum.md)，這條路線降低首次建核的頻譜容量需求。
