# 彈性 resolvent 的等價分解原型

2026-10-24。`tools/replay_resolvent.py` 目前用已完成的 BEM 頻譜核對新變數，
沒有以擬合取代彈性體，也沒有縮短已發生的非零記憶。
完整 event-16、可擴展 adjoint 與冷建立效能仍待驗證。

令 q 是靜態 lift 周圍的接觸 jumps／platen 擾動，Z 是有限體動態 impedance，
C 是原 kn／kt 乘以接觸線權重的彈性接觸矩陣，K 是 platen 彈簧。
把接觸力寫成線性部分加上修正：

```text
A(s) = Z(s) + C + K, L(s) = inverse(A(s))
chi(q,state) = Fc(q,state) - Fc0 - C*q
q_el = L * f_ramp
q = q_el + L * (f_hold - chi).
```

f_ramp 是持續原速率的 actuator 載入，f_hold 是原 coverage hold 相對它的偏差。
q_el 包含完整啟動、波傳與有限體反射；不是逐步重算的靜態平衡。
在閉合／黏著、狀態未變更且尚未 hold 時，修正源恆為零。
此時不配置線上修正歷史；第一次非零後保留全部項。
若事先給定的修正 horizon 不足，程式拒絕繼續，不會裁掉較舊資料。

每步仍求同一個 previous-state 摩擦平衡根。給定過去修正 h，使用
`past_virtual = f_ramp - A0*(q_el+h)` 接回現有局部 JAX Newton，
接觸係數、penalties 與 coverage hold 均不改動。

## 已有證據與限制

- 非交換矩陣的 inverse/composition、兩種 NPY 格式、初始零源 rebase 及拒絕漏掉非零源的測試。
- h=80 mm／32 步強載入診斷：最大滑移差約 9.13e-11 mm，壓力／剪力差小於 4e-9 MPa；actuator 相同。
- 該診斷第 5 步開始修正，沒有刪掉已發生的非零歷史。它不是原 45 ms event-16。

FFT aliasing／roundoff 使這個數值反變換不保證逐位一致，故另外檢查
`L0*A0-I` 並保留物理欄位差。尚未將本原型的條件式 adjoint 接回正式版本。
尤其 kn／kt 識別若把 C 固定為基準值，必須保留 chi 在破裂前的參數導數，
不能因基準修正為零而直接把這些導數也設為零。

目前先從既有完整頻譜建立 L，因此不據此宣稱降低冷核心計算時間。
後續要把彈性基準的向量頻譜與較短的修正矩陣歷史分開建立，並一併核對轉置傳播。
這是目前的實作方向，還不是已量測的速度改善。

```sh
FDBEM_CORES=1 bem/run.sh resolvent-replay bem/outputs/galerkin-disk-native-cli32 \
  --output bem/outputs/resolvent-native32-current
```

輸入須有完成標記與 SHA-256 的磁碟頻譜及相符的權重快取。
原頻譜只以唯讀方式分批讀取，新 L 頻譜與軌跡另存指定輸出資料夾。
