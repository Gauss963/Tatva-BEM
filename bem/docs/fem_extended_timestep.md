# 從 dt/4 延伸 FEM 時間步長控制

[原破裂後窗口](fem_postrupture_timestep.md)的 dt/2 對 dt/4 pressure／signed shear
時空 P1 L2 差為 17.68%／10.05%，尚未通過收斂診斷。
新工具從同一個原 checkpoint 與共同接受步，延伸到 dt/8、dt/16。
它保留原 h2／end 0.015625 mm 網格、材料、penalty、摩擦律、載入停止判據與全界面端點。
目前沒有這兩個完整長窗口的解差或收斂結論，也尚未證明完整 FEM／BEM 優勢。

## 數值來源與重播

`extend_fem_timestep_window.py` 以固定 SHA 的原 window driver 建立 mesh／設定／共同狀態，
在原 `records=[]` 之前返回 locals；原數值 setup 內容不改。
時間積分仍使用 archived Tatva 的原 step／scan closure，只以 model dt 建立各級時間步。
half-step velocity 依同一個原接受 kick 重新定相，維持相同 integer-time velocity。

新的 power-of-two loading schedule 在 factor 1／2／4 與舊工具 bitwise 相同，
factor 8／16 的所有巢狀共用端點也保持 bit patterns。
完整 dt/4 既有結果只在檢查其來源、checkpoint、輸出 SHA 並通過新驅動器重播後重用。
每次 invocation 都重新重播 6000 個原步長區間：
全部 24000 個 float64 scalar rows 與兩個共用時刻的八個完整接觸場須 bitwise 相同。
實際細網格的第一次重播已通過；真正中斷／恢復與不中斷 dt/8 的完整 pilot 對照仍在執行。

## 可恢復的接受狀態

`fem_refinement_chunks.py` 在每個原共用保存時刻（及窗口末端）提交一個 chunk：

- 保存該段**每個 fine step 的 17 個 float64 scalars**。
- 保存 chunk 末端的全部 **12 個 carry arrays**，包含速度、摩擦記憶、停止旗標與原時鐘。
- 共用保存時刻另保留原八個 float64 contact fields；不降低原接觸輸出密度。
- 數值 payload hash、前一個 chunk 的檔案 hash、factor、原步數與來源 manifest 綁定。
- NPZ 寫入、fsync、atomic rename、directory fsync 後才視為 committed。

resume 檢查完整連續 hash chain、shape／dtype／finite values、來源與 runtime，
只從完整的已接受 chunk 末端恢復。不完整 pending file 不視為 committed，也不自動刪除。
最終 export 需與 committed chunks 相符，並有獨立 receipt。
這裡的 accepted-state retention 指 **chunk 末端**，沒有儲存每個 fine step 的完整 bulk state。
summary 中的 `every_fine_scalar_and_accepted_state_retained` 也採此範圍。

26 項 unit controls 通過，包括舊 window 的既有測試、loading bits、所有 carry／scalar
的真正檔案恢復、缺少 row／grid endpoint、資料變動／不連續、重複寫入拒絕及原始時鐘。
它們驗證 driver／儲存／比較語義，不能代替完整 fine FEM 物理解與收斂證據。

## 執行與比較

```sh
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
JAX_ENABLE_X64=true python -B bem/tools/extend_fem_timestep_window.py \
  bem/outputs/fem-window-postrupture-study-20261030/window \
  --output bem/outputs/fem-window-finer-fresh

# 明確恢復同一個 output；來源、runtime 及數值參數必須一致。
PYTHONPATH=bem/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
JAX_ENABLE_X64=true python -B bem/tools/extend_fem_timestep_window.py \
  bem/outputs/fem-window-postrupture-study-20261030/window \
  --output bem/outputs/fem-window-finer-fresh --resume

python -B bem/tools/compare_extended_fem_windows.py \
  bem/outputs/fem-window-postrupture-study-20261030/window \
  bem/outputs/fem-window-finer-fresh \
  --output bem/outputs/fem-window-finer-comparison.json
```

預設保持原 230000 個原步長區間，約 440 µs；dt/8／dt/16 分別前進
1840000／3680000 個 fine steps。`--original-steps 6000 --factors 8` 僅供明確標記的短 pilot。
輸出 preflight 計入所有 scalar chunks、final export 與各 chunk carry 的 logical bytes，
預設 4 GiB array budget，另保留 1 GiB filesystem free-space reserve；不是硬 RSS 上限。
每次 invocation 與中斷的外層成本均保留，沒有未知終止時間的 attempt 不能補造總耗時。

本機執行仍需四核心預算與其他 numerical jobs 串行；外層控制會先核對原 BEM PID／start time，
暫停 producer 與 quota controller，結束後恢復相同程序。這段暫停仍計入其 wall cost。

比較工具保留全部原 contact endpoints、相同 saved clocks，並使用每個 fine scalar step。
dt/8 對 dt/16 的六個接觸場須在時空 P1 L2 與 final shared-time P1 L2 均小於 1%，
才通過**局部診斷**。同時報告 dt/4 差異、原 first-passage brackets、載入停止時間及 scalar 波形。
不做 fitted time shift、濾波、端點移除或 Richardson 外插。
即使此窗口通過，也仍缺完整事件時間／空間收斂與獨立 BEM 對照，原目標保持未完成。
