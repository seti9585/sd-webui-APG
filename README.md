# sd-webui-APG

**EN** | [日本語](#日本語)

Pre-CFG guidance extension for Stable Diffusion WebUI (Forge-based).
Decomposes the CFG guidance vector into components parallel and orthogonal to the conditional prediction, then down-weights the parallel component — the main source of oversaturation and artifacts at high guidance scales.

Paper: [arXiv:2410.02416](https://arxiv.org/abs/2410.02416) (ICLR 2025) — "Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models"

> When `Eta = 1`, `Norm Threshold = 0`, `Momentum = 0`, the result is identical to standard CFG.

---

## Installation

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-APG
```

---

## Parameters

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| Eta            | 0.00 | Fraction of the parallel component kept. `0` removes it entirely (paper-recommended default, strongest de-saturation). `1` keeps it fully, which makes the projection an identity operation. |
| Norm Threshold | 15.0 | Per-sample L2 clamp on the guidance vector. `0` disables the clamp. The paper's Stable Diffusion XL setting is `15` (Table 10). |
| Momentum       | 0.00 | Coefficient of the running average of the guidance vector across model evaluations. `0` disables it (default). The paper's experiments use negative values such as `-0.5`. **Off by default — see the note below.** |

Eta is the primary control. Raising it from 0 adds saturation back; lowering Norm Threshold below the guidance norm begins to cap single-step magnitude; Momentum smooths abrupt evaluation-to-evaluation changes but interacts with the sampler (see below).

### Suggested starting points (APG alone)

Based on an Eta sweep at CFG 7 (SDXL-family, TDE Sampler `euler`, AYS, single-variable A/B):

- Most of Eta's visible effect is concentrated in the `0.0 → 0.1` range; `0.2` adds a little more, and `0.3` and above show diminishing returns. This matches the linear form `update = orthogonal + Eta × parallel`.
- A practical everyday starting point is **Eta 0.2 / Norm Threshold 15 / Momentum 0** (saturation restored, safe). Lower Eta toward `0.1` for a more restrained, higher-contrast look.
- At the CFG 7 tested here the guidance norm rarely exceeds 15, so Norm Threshold is effectively an inert safety valve; drop it toward `10` only when you specifically want to cap single-step magnitude. At much higher CFG the clamp becomes active and its effect grows.

---

## Momentum and ODE samplers

Momentum keeps a running average of the guidance vector **across model evaluations**, not across visible sampling steps. This carried state is why it is disabled by default:

- The convergence order of Runge–Kutta and adaptive-step solvers assumes the right-hand side depends only on `(x, sigma)`. Eta and Norm Threshold are stateless per-evaluation transforms and preserve that assumption; Momentum does not.
- A multi-stage solver such as RK Sampler `fe_kutta4` evaluates four stages per step, so the running average updates four times per step, decaying older contributions faster than a one-evaluation-per-step solver would. The same Momentum value therefore behaves differently across samplers.
- Adaptive-step solvers reject and re-try steps; the rejected evaluations still accumulate, and a re-try at a temporarily larger sigma also trips the sigma-increase reset. Both make the effective Momentum depend on the tolerance settings.

If you use Momentum, prefer a one-evaluation-per-step fixed-step solver (e.g. `fe_euler1`, `euler`) and a value near `-0.5`. Otherwise leave it at `0`, where APG is a pure stateless per-evaluation transform.

---

## Algorithm

```
diff       = cond − uncond                          # denoised (x0) space
diff       = momentum_buffer.update(diff)           # optional (Momentum ≠ 0)
diff       = diff × min(1, norm_threshold / ‖diff‖) # optional (Norm Threshold > 0)
parallel   = project diff onto cond
orthogonal = diff − parallel
update     = orthogonal + Eta × parallel
output     = cond + (cfg_scale − 1) × update
```

Norm and projection reductions run over all non-batch dimensions (per sample). Projection is computed in double precision, then cast back.

When `Eta = 1`, `Norm Threshold = 0`, `Momentum = 0`, `update = diff = cond − uncond` and the output reduces to `uncond + cfg_scale × (cond − uncond)` — bit-for-bit standard CFG. This makes a clean fixed-seed A/B baseline.

### Differences from the ComfyUI built-in APG node

This port follows the paper's Algorithm 1, and two choices differ from ComfyUI's `comfy_extras/nodes_apg.py`:

- **Final combination.** This extension uses `cond + (cfg_scale − 1) × update`, matching the paper. The ComfyUI node effectively yields `cond + cfg_scale × update`, one guidance unit stronger, so its neutral settings do not reduce to standard CFG. The paper form is used here specifically so the neutral A/B baseline holds.
- **Reduction dimensions.** Norm and projection reduce over all non-batch dimensions (`range(1, ndim)`) instead of a fixed `dim=[-1,-2,-3]`. For 4-D SDXL latents `(B, C, H, W)` this is identical to the paper. For 5-D latents `(B, C, T, H, W)` it keeps the paper's per-sample semantics instead of silently becoming per-channel. HuggingFace diffusers' `AdaptiveProjectedGuidance` makes the same choice.

---

## Backend-adaptive hooking

| Backend | Hook | Mechanism |
| ------- | ---- | --------- |
| reForge / Forge Classic | Pre-CFG | The uncond slot of `conds_out` is overwritten with `cond − update`. The backend's standard CFG step then produces `cond + (cfg_scale − 1) × update` for any CFG scale — the write-back itself is scale-independent. |
| Forge Neo | Post-CFG | Forge Neo's pre-CFG runs before model evaluation, so predictions are not available there. The final prediction is recomputed directly as `cond + (cfg_scale − 1) × update`, priority-ordered in the post-CFG list. |

A fresh momentum buffer is created for every sampling pass (txt2img and Hires.fix get independent state), with a secondary reset when sigma increases between calls.

On reForge, the first-step `cond_scale` was verified to equal the configured CFG value (no `cond_scale = 1.0` first-step quirk); the write-back is scale-independent regardless.

---

## Compatibility with other extensions

APG (Pre-CFG, `sorting_priority = 14.5`) sits at the end of the pre-CFG chain, just before the CFG core:

```
TCFG (13.0) → SkimmedCFG (14.0) → DifferenceCFG (14.2) → APG (14.5) → CFG core → CFGZeroStar (15.0) → MaHiRo (15.5)
```

This placement matches APG's role as the final reshaping of the guidance vector before CFG applies it. On Forge Neo, TCFG's damped uncond is read from `model_options["_tcfg_damped_uncond"]` when TCFG ran earlier in the same post-CFG list, so the two compose correctly.

When stacking multiple CFG-axis extensions, keep the session CFG within a moderate range to avoid cumulative correction breakdown.

### Note on high CFG

APG is designed to address exactly the oversaturation and artifacts that high guidance scales produce, so it is a natural fit for high-CFG workflows. At high CFG the guidance norm grows, so Norm Threshold (default 15) starts to clamp actively and its effect becomes visible; the best Eta may also differ from the low-CFG case. A separate sweep at your working CFG is worthwhile.

---

## Implementation note — `process_before_every_sampling()`

Forge-based WebUIs rebuild `forge_objects.unet` between `process()` and the actual sampling start.
Any hook registered in `process()` is silently discarded when this rebuild occurs.

This extension registers its hook in `process_before_every_sampling()`, where `forge_objects.unet` is already the same object the sampler will reference.
Metadata for PNG Info is written separately in `process()` so `create_infotext` captures it. The `APG Eta` key is written only when the extension is active, so its presence on the read side doubles as the enable marker for the PNG Info round-trip.

---

## Tested environments

- reForge (Python 3.10) — SDXL-family models; txt2img confirmed, first-step `cond_scale` verified.
- Forge Neo (Python 3.12) — post-CFG path.

Not compatible with A1111 (`set_model_sampler_pre_cfg_function` is Forge-backend only).

---

---

# 日本語

**[English](#sd-webui-apg)** | 日本語

Forge 系 WebUI 向け Pre-CFG ガイダンス拡張機能。
CFG ガイダンスベクトルを、条件付き予測に対して平行な成分と直交な成分に分解し、平行成分を減衰させます。平行成分は高いガイダンススケールで生じる過飽和とアーティファクトの主因です。

論文: [arXiv:2410.02416](https://arxiv.org/abs/2410.02416)（ICLR 2025）— "Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models"

> `Eta = 1`・`Norm Threshold = 0`・`Momentum = 0` のとき、通常 CFG と完全に等価です。

---

## インストール

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-APG
```

---

## パラメータ

| パラメータ | 既定値 | 説明 |
| --- | --- | --- |
| Eta            | 0.00 | 平行成分を残す割合。`0` で完全に除去（論文推奨の既定値、脱飽和が最も強い）。`1` で完全に残し、射影が恒等変換になる。 |
| Norm Threshold | 15.0 | ガイダンスベクトルへのサンプルごとの L2 クランプ。`0` で無効。論文の Stable Diffusion XL 設定は `15`（Table 10）。 |
| Momentum       | 0.00 | モデル評価をまたぐガイダンスベクトルの移動平均係数。`0` で無効（既定）。論文の実験では `-0.5` などの負値を使用。**既定では無効 — 下記の注記を参照。** |

Eta が主要な制御軸です。0 から上げると彩度が戻り、Norm Threshold をガイダンスノルムより下げると単一ステップの大きさを抑え始め、Momentum は評価間の急変を平滑化しますがサンプラーと相互作用します（下記参照）。

### 目安の初期値（APG 単体）

CFG 7（SDXL 系、TDE Sampler `euler`、AYS、単一変数 A/B）での Eta スイープに基づく目安：

- Eta の見た目の効果は大半が `0.0 → 0.1` の区間に集中し、`0.2` で少し追加、`0.3` 以上は逓減します。これは線形の式 `update = 直交 + Eta × 平行` の通りの挙動です。
- 実用的な常用の出発点は **Eta 0.2 / Norm Threshold 15 / Momentum 0**（彩度が戻り、安全）。より締まった高コントラスト寄りにするなら Eta を `0.1` へ下げます。
- ここでテストした CFG 7 ではガイダンスノルムが 15 を超える場面が少なく、Norm Threshold は実質的に無発動の安全弁です。単一ステップの大きさを明示的に抑えたいときだけ `10` 前後まで下げてください。より高い CFG ではクランプが発動し、効果が大きくなります。

---

## Momentum と ODE サンプラー

Momentum は、可視のサンプリングステップではなく **モデル評価** をまたいでガイダンスベクトルの移動平均を保持します。この状態の持ち越しが、既定で無効にしている理由です：

- Runge–Kutta 法や可変ステップ法の収束次数は、右辺が `(x, sigma)` のみに依存することを前提とします。Eta と Norm Threshold は評価ごとのステートレスな変換でこの前提を保ちますが、Momentum は保ちません。
- RK Sampler `fe_kutta4` のような多段ソルバーは 1 ステップあたり 4 回評価するため、移動平均は 1 ステップに 4 回更新され、1 評価/1 ステップのソルバーより古い寄与が速く減衰します。同じ Momentum 値でもサンプラーによって効き方が変わります。
- 可変ステップ法はステップを棄却・再試行します。棄却された評価も蓄積され、一時的に大きい sigma での再試行は sigma 増加リセットも誤発火させます。どちらも実効的な Momentum を許容誤差設定に依存させます。

Momentum を使う場合は、1 評価/1 ステップの固定ステップソルバー（例：`fe_euler1`・`euler`）で `-0.5` 前後を推奨します。それ以外では `0` のままにしてください。`0` のとき APG は純粋にステートレスな評価ごとの変換になります。

---

## アルゴリズム

```
diff       = cond − uncond                          # denoised (x0) 空間
diff       = momentum_buffer.update(diff)           # 任意（Momentum ≠ 0）
diff       = diff × min(1, norm_threshold / ‖diff‖) # 任意（Norm Threshold > 0）
parallel   = diff を cond 方向へ射影
orthogonal = diff − parallel
update     = orthogonal + Eta × parallel
output     = cond + (cfg_scale − 1) × update
```

ノルムと射影の集約は、バッチ以外の全次元（サンプルごと）で行います。射影は倍精度で計算してから元の型に戻します。

`Eta = 1`・`Norm Threshold = 0`・`Momentum = 0` のとき `update = diff = cond − uncond` となり、出力は `uncond + cfg_scale × (cond − uncond)`、すなわち通常 CFG とビット同一になります。固定シードの A/B 基準として利用できます。

### ComfyUI ビルトイン APG ノードとの差異

本移植は論文の Algorithm 1 に従っており、ComfyUI の `comfy_extras/nodes_apg.py` と 2 点で異なります：

- **最終合成。** 本拡張は論文に合わせて `cond + (cfg_scale − 1) × update` を使います。ComfyUI ノードは実質的に `cond + cfg_scale × update` となり、ガイダンス 1 単位分強く、中立設定でも通常 CFG に一致しません。中立の A/B 基準を成立させるため、本移植では論文の形を採用しています。
- **集約次元。** ノルムと射影を固定の `dim=[-1,-2,-3]` ではなく、バッチ以外の全次元（`range(1, ndim)`）で集約します。4 次元の SDXL 潜在 `(B, C, H, W)` では論文と同一です。5 次元潜在 `(B, C, T, H, W)` ではチャンネルごとに変化させず、論文のサンプルごとの意味を保ちます。HuggingFace diffusers の `AdaptiveProjectedGuidance` も同じ選択をしています。

---

## バックエンド適応フック

| バックエンド | フック | 仕組み |
| --- | --- | --- |
| reForge / Forge Classic | Pre-CFG | `conds_out` の uncond スロットを `cond − update` で上書きします。バックエンドの標準 CFG ステップが、任意の CFG スケールで `cond + (cfg_scale − 1) × update` を生成します。上書き自体はスケールに依存しません。 |
| Forge Neo | Post-CFG | Forge Neo の pre-CFG はモデル評価の前に走るため予測が利用できません。最終予測を `cond + (cfg_scale − 1) × update` として直接再計算し、post-CFG リスト内で優先度順に配置します。 |

Momentum バッファはサンプリングパスごとに新規作成され（txt2img と Hires.fix は独立した状態を持つ）、呼び出し間で sigma が増加した際の二次的なリセットも備えます。

reForge では、初回ステップの `cond_scale` が設定した CFG 値と一致することを確認済みです（`cond_scale = 1.0` になる初回ステップの癖はなし）。上書きはいずれにせよスケール非依存です。

---

## 他拡張との併用

APG（Pre-CFG、`sorting_priority = 14.5`）は pre-CFG チェーンの末尾、CFG コアの直前に位置します：

```
TCFG (13.0) → SkimmedCFG (14.0) → DifferenceCFG (14.2) → APG (14.5) → CFG コア → CFGZeroStar (15.0) → MaHiRo (15.5)
```

この配置は、CFG が適用する直前にガイダンスベクトルを最終整形するという APG の役割に合致します。Forge Neo では、同じ post-CFG リスト内で TCFG が先に走った場合、TCFG の減衰済み uncond を `model_options["_tcfg_damped_uncond"]` から読み取るため、両者は正しく合成されます。

複数の CFG 軸拡張を重ねる場合は、セッション CFG を穏当な範囲に抑え、補正の累積破綻を避けてください。

### 高 CFG について

APG は、高いガイダンススケールが生む過飽和とアーティファクトそのものに対処する設計なので、高 CFG ワークフローと自然に相性が良いです。高 CFG ではガイダンスノルムが大きくなるため、Norm Threshold（既定 15）が実際にクランプを始め、効果が可視化されます。最適な Eta も低 CFG の場合とは異なる可能性があります。運用する CFG での個別のスイープを推奨します。

---

## 実装上の注意点 — `process_before_every_sampling()` の使用について

Forge 系 WebUI は `process()` の実行後、サンプリング開始前に `forge_objects.unet` を再構築します。
そのため `process()` 内で登録したフックは再構築時に消えてしまい、サンプリング中に一切呼ばれません。

本拡張はフック登録を `process_before_every_sampling()` で行っています。このタイミングでは `forge_objects.unet` がサンプラーの参照先と同一オブジェクトです。
PNG Info 用のメタデータは `create_infotext` が拾えるよう `process()` で別途書き込みます。`APG Eta` キーは拡張が有効なときのみ書かれるため、読み取り側ではその存在が PNG Info 往復の有効化マーカーを兼ねます。

---

## 動作確認環境

- reForge（Python 3.10）— SDXL 系モデル。txt2img 確認済み、初回ステップの `cond_scale` を検証済み。
- Forge Neo（Python 3.12）— post-CFG 経路。

A1111 非対応（`set_model_sampler_pre_cfg_function` は Forge バックエンド専用）。

---

## ライセンス・典拠

Based on: [arXiv:2410.02416](https://arxiv.org/abs/2410.02416) "Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models" (ICLR 2025)

Reference implementations consulted: the ComfyUI built-in APG node (`comfy_extras/nodes_apg.py`) and **Shiba-2-shiba**'s APGForge implementation for Forge Classic. This extension is written from the paper above; the pointers that made it knowable are gratefully acknowledged.

Inspired by **Shiba-2-shiba**'s note articles on CFG-related ComfyUI nodes:
[ComfyUIのCFG関連の4ノードの勉強＠APG, TCFG, Fresca, Mahiroノードについて](https://note.com/gentle_murre488/n/nc709aac794bc)

本拡張機能は、**Shiba-2-shiba** さんの note 記事「[ComfyUIのCFG関連の4ノードの勉強＠APG, TCFG, Fresca, Mahiroノードについて](https://note.com/gentle_murre488/n/nc709aac794bc)」から着想を得ています。原実装として ComfyUI ビルトイン APG ノードおよび Shiba-2-shiba さんの Forge Classic 向け APGForge 実装を参照しました。本拡張は上記論文に基づいて記述しています。
