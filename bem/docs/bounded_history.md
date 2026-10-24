# 限制 FFT 快取，保留完整因果歷史

2026-10-24。這項實驗讓 resolvent 的 forward 與 transpose history 可指定保留快取容量，
支援較細時間步的資源規劃。它不刪長尾、不擬合權重，也不改摩擦律。
[數值、來源與指紋](bounded_history_results.json) 包含通過的比較與仍未通過的條件。

## 方法與資源估計

沿用原 dyadic source/target 配對及小區塊的直接加總。
按容量預先選擇反覆使用的小層級；每個完整 kernel FFT 需要 `(b+1)*d*d*16` bytes。
大層級改為每次逐列計算 FFT，仍把每個 source 的作用加到所有所需 target。
固定選擇小層級，避免在時間推進中反覆淘汰與重建它們。
零容量會串流所有 FFT 層級；當步權重與完整時間權重皆保留。

同一實作作用在轉置權重上，保留 JAX 局部 VJP、隱式根微分、
完整反向卷積與初始 elastic prefix 的 cotangent。
zero-prefix 的檢查沿用原 helper，只替換其私有卷積 factory；套件 globals 不變。

`--fft-cache-mb` 以 1024² bytes 為一單位，限制的是**保留的 kernel FFT**，
不是全部 RAM。預估另含列 FFT／padding、完整軌跡與 adjoint vectors。
prepared kernel 使用唯讀磁碟權重；legacy replay 仍在 RAM 產生 prefix，須額外計入。
OS file pages、JAX／BLAS cache 不能當成此估計提供的硬 RSS 保證。
native 建核亦增加時間權重與完整啟動的磁碟預留檢查，避免頻譜先耗盡它們的空間；
這是逐次 free-space 檢查，不是隔離其他程序的磁碟 quota。

## 完整粗網格：4 us、45 ms

使用同一已驗證的壓縮 L kernel：71 retained DOFs、11250 步、5500 correction lags。
這裡驗證新的線上 history，沒有重新宣稱 fresh native 取樣已完成完整事件。

|設定|實際保留 kernel FFT|觀測程序 peak RSS|
|---|---:|---:|
|原快取規則|未另加指定容量限制|787.95 MiB|
|指定 0 MiB|0|627.92 MiB|
|指定 8 MiB|7,904,288 bytes，約 7.54 MiB|649.75 MiB|

0／8 MiB 的壓力、剪力、塑性與累積滑移、actuator、platen、contact jumps，
以及五參數、elastic prefix、每步 actuator、兩種初始狀態 cotangent，
均與原快取規則**逐位元相同**。四種精確接觸分支變更為零。
對獨立原 Z forward sensitivity，五參數最大相對差約 2.23e-6，通過原容差。
後續補做既有直接 L 的完整 actuator／初始狀態比較時，初始狀態通過，
但 actuator index 1724 有一筆未通過原容差；[近遠場實驗](mixed_quadrature.md)
記錄該控制組及獨立 forward JVP。快取規則之間的逐位元等價仍成立。

另將先前 fresh native 建立的 512 步診斷 kernel 與 0 MiB 快取組合，
物理欄位及所有支援的 cotangent 也與原規則逐位元相同，並通過獨立原 Z guard。

程序依序執行，但同時有 fine2 建核，因此 RSS／耗時不是受控效能比較。
此項只證明容量限制與數值等價，沒有宣稱比 FEM 更快。

## 較細網格 adjoint：275 retained DOFs

完整 4 us／45 ms、5000 correction lags 的 legacy L 軌跡，也分別使用原規則及
128 MiB 上限重算 adjoint。實際保留 118,580,000 bytes，約 113.09 MiB。
五參數、prefix、actuator 與初始狀態 cotangent 均逐位元相同，故容量限制沒有額外梯度誤差。

對獨立原 Z adjoint，五參數最大相對差約 1.80e-7，初始狀態亦通過原容差。
但**完整逐點 AD guard 仍未通過**：actuator 第 2393 步（9.572 ms）
參考值 -0.0002668976121、L 結果 -0.0002669033059，絕對差 5.69e-9，
超過 `atol=1e-12, rtol=1e-5` 的門檻。兩種快取皆有同一個失敗點，CLI 均以 exit 1 結束。
這個差異屬於既有 L／Z 對照；並未放寬 guard 或標成所有梯度通過。
該 legacy L 軌跡先前的嚴格 all-field replay guard 亦未通過，仍保留為待改進項目。

AD 範圍是固定 bulk、lift、保存載入與 active branches，尚不含材料、幾何與 hold 時間的導數。
完整案例與 FEM 的物理解差距仍在，這項儲存改良不代表事件重現已完成。

## 使用

```sh
FDBEM_CORES=4 bem/run.sh resolvent-simulate PREPARED_KERNEL \
  --fft-cache-mb 128 --max-memory-mb 8192 --output NEW_RUN

# 指定獨立原 Z 參考；省略第二個 run 則只計算 standalone adjoint。
FDBEM_CORES=4 bem/run.sh resolvent-adjoint ORIGINAL_Z_RUN NEW_RUN \
  --reference-gradient ORIGINAL_GRADIENT_JSON \
  --fft-cache-mb 128 --max-memory-mb 8192
```

省略 `--fft-cache-mb` 時維持原快取規則。`--fft-rows` 預設 4，可調整每次串流的輸出列數。
容量、列數、保留峰值與來源 SHA-256 都會記錄。
非對稱權重、正向／轉置、非二次冪長度、磁碟資料與零前綴／horizon guards 有獨立測試。
原 2 us／45 ms 四-worker 建核繼續使用原 backend；尚未將較細步長標示為已完成驗證。
