# 與 Tatva 一致的 healed 靜態接觸初始化

2026-10-27 核對 archived Tatva `933ae1f` 的
`tatva/pmma/static_normal.py:57`、`dynamics.py:3724` 後，新增顯式控制：

```sh
./bem/run.sh refined-contact-simulate KERNEL \
  --restart-seed SEED --subdivisions 4 --constitutive-substeps 64 \
  --kinetic-root-seed --static-initialization fem-active-set --output NEW_OUTPUT
```

FEM 的 restart plastic slip 是靜態 stick 分支的無應力偏移；滑動點會先以
healed Coulomb 強度重算偏移，接著把 weakening 記憶設為 healed 值。
本 event 的 healed 值為零。直接固定所有偏移只在靜態解已符合強度時才等價。

固定 bulk 的原始 Γ1、Γ2 控制皆符合強度；Γ4、Γ8 在 y=479.5 mm 的 frozen
試算分別超過強度約 5.36、13.34 MPa，完整體平衡 residual 則約 1e-10 MPa。
這是初始強度條件失敗，不能藉放鬆 residual tolerance 或增加動態 Newton 次數處理。

`fem_static_contact.py` 依 Tatva 的開閉、stick/sliding、滑動方向 active set
組裝非對稱接觸切線，保留正權重及功共軛 pullback。它使用固定 healed 強度，
不在靜態載入中演化 slip weakening。得到偏移後，再使用原 frozen initializer
獨立重算並檢查原 full-body 平衡、trace、強度，容許誤差不變。

輸入 seed 不改寫。輸出記錄滑動／開啟點數、每次 active set、最大偏移修正及
memory 是否不變；動態求解明確從 `lift.state` 開始。預設仍為 `preserve-offset`，
過去實驗的物理定義保持可重現。兩種初始化必須分別標示，不能混稱純積分加密。

單點飽和靜態方程與開啟／stick 控制，加上既有 refined-contact 測試，本機
15 tests passed。完整 Γ4/Γ8 事件與 FEM 比較尚待實跑；尚未證明同精度優勢。
目前 AD 仍以完成的 lift 與初態固定為條件，沒有宣稱微分了這個 active-set lift。

另有 `fem_static_contact_ad.py` 原型：固定已選定的開閉／滑動集合，使用 JAX
微分收斂的線性平衡及輸出 plastic offset，避免反傳 active-set 迭代。
新增正負滑動及 stick 的五參數有限差分與 JVP/VJP 對照，交由 CI 核對。
它尚未接入完整 transient gradient；接續仍須傳遞初始位移、bulk force、
plastic offset 與 initial contact force 對所有後續歷史的影響。
