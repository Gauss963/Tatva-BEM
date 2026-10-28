# 完整 graded FEM 參考與四核心執行紀錄

2026-10-28 的[狀態快照](reference_first_results.json)。這是執行中的工作紀錄；
完整 event-16 的 FEM/BEM 一致性、收斂與效能優勢仍未通過。

## 中斷與 checkpoint 接續

12:24 UTC 再檢查時，原 FEM、BEM、兩個 supervisor 與 CPU 控制器的 PID 及
exec handles 都已不存在；沒有正常退出紀錄，原因尚未確認，主機並未重啟。
最後可證實存活的 CPU 紀錄在 11:14:38 UTC；不能由原本的狀態檔推論仍在執行。
原先 `finally` 恢復同一 PID 的計畫也已無法執行。

已備份並驗證 partial HDF5、checkpoint、manifest、resolved config、projected restart
與 invocation ledger。checkpoint 是總時間 2.515059658 ms，即 normal phase 已完成、
shear phase 已推進約 0.515060 ms；其後尚未 checkpoint 的少量推進需重算。
12:33:37 UTC 曾用原 runner 的 `--resume` 嘗試接續，但 12:34 時讀寫壓縮資料失敗。
完整 chunk 稽核找到 `interface_high_rate/history` 的第一個 chunk 已損毀；
低頻 `history` 雖可解壓，時間欄卻不正確。float64 接觸場與 checkpoint 仍可讀，
但這份 HDF5 不能當成完整有效的參考，也沒有填補或猜測遺失的歷程值。
先嘗試的 launchd wrapper 在 Python 目錄讀取時等待，尚未啟動數值子程序；已移除該
job，改用繼承目前桌面存取環境的 detached wrapper，並保留兩次啟動紀錄。

新增 [`run_checkpointed_fem_reference.py`](../tools/run_checkpointed_fem_reference.py)，
從原始初態重播到既有 checkpoint，要求全部 12 個 state arrays 逐位元相同，
才繼續完整原時間範圍。它在原 output checkpoint 分支增加一次性的 checkpoint
請求，並包裝輸出 observer／checkpoint writer；移除這兩處 output AST 增補後，
其餘 AST 必須與 pinned FEM 完全相同。計算方程、接觸、步長與原 source files 均未改動。
capture 放在兩個輸出 writer 與 normal handoff 之後，保留完整 state 與所有已完成輸出。

每次 checkpoint 同步保存已 flush 的 HDF5 與 NPZ，重新讀取複本的壓縮資料、
純量與 float64 時鐘，核對兩檔 metadata 與 SHA256 後，才原子更新 snapshot pointer。
保留最近兩個成對快照；寫入中的主檔若再損毀，可以使用先前完整複本。
這會增加真實 I/O 成本，會計入計時，並非數值速度提升。

實際小型 pinned FEM 控制已通過：12 個 state arrays 與接續完成後的 **65 個 datasets**
全部逐位元相同。另有 17 個單元測試涵蓋壓縮損毀後的 snapshot 保留、靜默時間欄
損毀拒絕、完整狀態核對、capture 順序及 CPU 控制。這些不代表完整 graded event 已通過。
12:56:56 UTC 已啟動新的 graded FEM 重播／完整接續工作，數值程序與 CPU 控制器均已確認存活。
新參考輸出為 `bem/outputs/fem-event16-radau-study-graded-h2-end015625-4c-f64-replayed`；
目前尚在重播前段，graded checkpoint 的 12-array 比較與完整 45 ms 尚待完成。

BEM 的 256 個已寫入頻點仍在磁碟，1,154,912,256 個已保存資料 bytes 的 SHA256 為
`f3802c5c698bed22e55be770bb4672ee9ec86d50c9ab24f1e9f1fc982c905801`。
記憶體中的插值狀態已消失，原程序無法直接恢復。後續新增的
[磁碟接續功能](radau_sampling_checkpoint.md) 已通過小型完整建核中斷控制，
並只讀驗證實際的 256 個頻點；完整細網格建核仍需在 FEM 之後接續。

前次 FEM invocation 沒有結束計時事件；不補造精確的結束秒數。
原 ledger、最後存活證據與本次接續時間分開保留，未完成的成本歸屬不能用來支持速度比。
以下保留中斷前的設定與排程紀錄，並非舊 PID 仍存活的宣告。

## 中斷前的順序與設定

原排程先取得獨立的完整 graded FEM 參考，再繼續同一個 fresh Radau 建核程序。
FEM 保留 2 ms normal phase、45 ms shear phase、原摩擦參數與載入政策，
背景網格 2 mm，接觸端點 0.015625 mm。計算與新增的接觸觀測值均為 float64。
自動穩定步長約 1.913 ns，總步數約 2457 萬；目前已開始輸出 normal-phase 進度。
完整 FEM 尚未完成，不能從部分輸出推算 rupture 的一致性。

fresh Radau 使用相同幾何、Q12、2 µs、完整 45 ms；其 bulk 有 265 個接觸節點，
含 platen 的 retained system 為 531 維。建核程序當時收到 SIGSTOP，
保留同一個 PID、partial files 及已計算頻點。等待前的 slow-branch 狀態是
195 次原始取樣、256/32769 個頻點已寫入，未丟棄資料。
原排程的 FEM 階段以 `finally` 恢復同一建核程序；程序整體中斷後這個清理未能執行。
這是依序安排必要工作，目標沒有暫停。

後續依序執行兩組完整接觸子步軌跡、完整原始 RK 因果算子稽核、六個接觸場的
FEM 對照與子步比較、signed incremental bulk port work，最後才是完整初始化 AD。
AD 的前置條件為六場的 FEM 差皆低於 5%、子步差皆低於 1%；空間與 macro-time
收斂、完整物理能量及同精度效能比較還有各自的檢查，這些百分比不是文獻定理。

兩個執行器分別凍結 66 與 58 個輸入檔，快照時全部 SHA256 相符。
數值程式使用 `ed57e44`；其 Python 3.11/3.12 CI 已通過。
新控制器與本說明沒有修改執行中凍結的數值程式。
所有新增輸出位於本專案與獨立的 `Tatva-BEM-cache`；原 Tatva checkout 維持只讀。

## 四執行緒與平均 CPU 配額

此 macOS 的四執行緒環境設定仍讓 FEM 平均使用約 4.65 個核心。
先嘗試的外部 `cpulimit` 沒有實際限制成功，已停止該控制器；它的來源、建置與
失敗量測保留在忽略的輸出目錄，沒有將第三方程式納入本套件。

新增 [`limit_process_cpu.py`](../tools/limit_process_cpu.py)，直接以 `ps` 的累積
程序 CPU 計時器控制 SIGSTOP/SIGCONT duty cycle，涵蓋仍在執行的執行緒。
它只使用 Python 標準函式庫；指定平均 3.85 個核心，預留監控開銷。
獨立 45.0145 秒量測記錄 173.29 CPU 秒，即平均 **3.84965 核心**。
這是平均 CPU 配額，並非 CPU affinity 或固定執行緒數；短時間可以超過四核心。
控制器不包含子程序的 CPU；本次 FEM 及已排程的重工作使用同一程序內的執行緒。

控制器要求 PID、啟動時間及 command fragment 三者相符，拒絕接管原先已停止的程序。
CPU credit 最多保留一個 work interval，避免長時間 I/O 等待後累積大量執行配額。
SIGTERM/SIGINT 會恢復它自己停止的目標；目標身分改變時不發送訊號。
必須先停止控制器，才可刻意暫停其目標；SIGKILL 無法執行恢復清理。
它是需要獨占目標程序控制權的本機研究工具，不是作業系統的硬資源隔離。

一般使用方式如下；PID 與啟動時間必須由呼叫者自己持有的程序取得，
同一目標同時只能有一個控制器：

```sh
python -B bem/tools/limit_process_cpu.py \
  --pid "$RUN_PID" --expected-start "$RUN_START" \
  --expected-command-fragment run_fem_reference.py \
  --cores 3.85 --output bem/outputs/my-cpu-quota
```

**FEM 前約半小時是在配額生效前執行。** 因此本次完整參考用於準確度對照，
不能標成「全程嚴格四核心」的獨立速度基準。配額生效時間及失敗量測均保留。

## 時間與驗證

FEM 的完整 wall interval 包含在暫停中的 BEM 建核 interval 內。
總成本會合併重疊的 UTC interval，並分列建核包含／扣除這段等待的時間，
避免把 FEM 重算兩次。先前已付出的 81293.978 秒建核成本仍保留；
不會透過忽略較早的嘗試宣稱總時間改善。

配額控制器有 CPU clock、變動用量的配額累積、PID 身分及中斷恢復測試；
獨立 CPU 計時器量測則檢查這台 macOS 上的實際效果。
完整數值檢查由上述凍結的工作依序執行，結果尚待完成。
