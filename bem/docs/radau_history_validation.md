# Radau 完整因果方程與條件式 adjoint

2026-10-28。[量測紀錄](radau_history_results.json) 與
[fresh Radau producer](raw_radau.md) 相互獨立保存，未把短程測試當成 event-16 重現。

`tools/audit_compact_radau_history.py` 由兩族完整 work-coordinate compliance
還原每個 contour point 的兩個實際 mode，分別求逆，再形成 original-operator
兩族係數。逐列 IFFT 保留到所需 horizon 的每個 lag，再作因果線性卷積。
Δ(z)=Δ0+zΔ1 所造成的瞬時 finite-DFT wrap 亦保留。
直接乘上未截除 padding 的頻譜會混入未來作用，這條捷徑沒有使用。

實際 h160／Q12 雙 block 的 128 步滑移診斷使用完整 FFT=65536 circle；
32769 個頻點、兩個 mode 全部求逆。最大 inverse-composition 殘差為 4.45e-12，
另外重新組裝八個原始 BEM mode，最大 solve-composition 殘差低於 2e-13。
所有保存 stage 的原方程最大殘差為 **9.4164e-8 MPa**，通過原 1e-7 MPa 門檻。
目前仍只檢查 128 步保存軌跡，並未檢查完整 45 ms fine 事件。

CLI 預設要求完整 prepared horizon。短程診斷必須明確給 `--allow-prefix`；
資源規劃、實際 horizon、來源 hashes 與完整方程結果分開記錄。
可用 `--remove-reconstructed-families` 在檢查後刪除新建、可重新生成的
original-operator 兩族檔案；來源 kernel 與 residual evidence 均保留。
這個 audit 共用 primal 的插值頻譜，驗證因果時間代數；獨立 BEM 抽查與
時間／空間收斂仍是另外的要求。

## JAX 的分工

`tools/radau_contact_adjoint.py` 對完整耦合 stage equilibrium 作隱式微分：
局部摩擦律使用 JAX JVP／VJP；每步解一次 tangent 或 transpose tangent 系統，
並經完整兩族 CQ 歷史傳遞。第一 stage 的試算狀態會進入 endpoint，但只有
endpoint 狀態與位移傳到下一步。每個實際 stage actuator 都有自己的 cotangent。
反向歷史共用兩 stage 的 kernel FFT；批次 forward JVP 再跨導數方向共用 FFT。
沒有刪除 primal 尚未滑動時的導數歷史，也不需記錄每次 Newton 迭代的計算圖。

實際上述雙 block 非線性診斷，五個接觸參數與兩個獨立 stage-loading 方向的
reverse／forward dot-product 最大絕對差為 **1.85e-14**。
單元控制另與重新組裝 original-Z RK-CQ 的有限差分比較，並涵蓋無快取與
有界快取的歷史路徑。這些是固定 bulk、初始 lift／狀態／實測力、reference
diagonal 與 affine startup 的**條件式**導數。尚未包括 Radau healed preload、
停止載入事件時間、材料或幾何導數；既有 BDF2 initialized AD 的驗證不能直接轉移。

## Fine 執行狀態

fresh fine h10／contact2／end0.015625、Q12、2 µs／45 ms 工作已在本機啟動，
raw 建構使用兩 workers，總預算四核心；其後排定兩組 constitutive 子步的完整事件。
先前尚未完成的 BDF2 工作已停止，所有部分頻譜均保留，先前通過的 1024 個矩陣
prefix hash 再次吻合。累計未完成建核成本為 **81293.98 s**，另列切換與修復紀錄
耗時，沒有將這些時間歸零。

這些數字不代表速度優勢。完整 fine event、原方程 audit、graded FEM 參考、
whole-interface 空間／時間誤差、macrostep／接觸積分收斂與 initialized Radau AD
仍需完成。原 `/Volumes/Gauss-T7/tatva` 維持唯讀。
