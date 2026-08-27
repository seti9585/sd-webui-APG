# sd-webui-APG

**EN** | [日本語](#日本語)

Adaptive Projected Guidance (APG) extension for Stable Diffusion WebUI (Forge-based).

Implementation of [**Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models**](https://arxiv.org/abs/2410.02416) (ICLR 2025), Algorithm 1.

> APG reduces the oversaturation and artifacts that appear at high CFG scales,
> letting you raise CFG for stronger prompt adherence without the usual
> color burn and contrast blowout.

---

## Features

- **Paper formulation**, not the ComfyUI node formulation — the neutral
  settings reduce to standard CFG **algebraically**. The ComfyUI node does
  not, at any setting. See [Neutral settings](#neutral-settings) for why this
  is not a bitwise identity.
- Works on **reForge / Forge Classic** (Pre-CFG hook) and **Forge Neo**
  (Post-CFG hook); the backend is detected automatically.
- Rank-agnostic projection — supports both 4-D SDXL latents and 5-D
  Anima / NextDiT latents.
- Composes with the rest of the guidance suite via priority insertion at `_sd_webui_priority = 14.5` — see [Composition with other extensions](#composition-with-other-extensions).
- XYZ Grid axes for all parameters.
- Generation parameters are embedded in PNG infotext for reproducibility.

---

## Installation

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-APG
```

> This extension relies on the Forge backend hook API.
> It is not available in A1111 (AUTOMATIC1111).

---

## How it works

APG decomposes the guidance vector into a component parallel to the
conditional prediction and a component orthogonal to it. The parallel
component is what pushes the latent along the direction it is already
heading — the main cause of oversaturation at high CFG. Scaling it down
keeps the semantic steering (orthogonal part) while removing the burn.

```
diff      = cond - uncond
diff      = momentum_buffer.update(diff)      (optional, beta != 0)
diff      = clamp L2 norm to norm_threshold   (optional, threshold > 0)
par, orth = project diff onto cond
update    = orth + eta * par
final     = cond + (cond_scale - 1) * update
```

---

## Parameters

| Control | Range | Default | Description |
|---|---|---|---|
| **Enable APG** | — | Off | Master switch. |
| **Eta** | 0.0 – 2.0 | 0.0 | How much of the parallel component to keep. `0` is the paper's recommended default; `1.0` keeps it fully, which disables the projection. Raise it if the result looks flat or desaturated. |
| **Norm Threshold** | 0.0 – 50.0 | 15.0 | Per-sample L2 clamp on the guidance vector. `0` disables. The paper uses `15` for SDXL. |
| **Momentum** | -1.5 – 1.0 | 0.0 | Running-average coefficient (beta). `0` disables. The paper uses negative values such as `-0.5`. **See the warning below before enabling.** |

### Neutral settings

```
Eta 1.0 / Norm Threshold 0 / Momentum 0
```

These reduce to standard CFG **algebraically**: with `Eta 1.0` the parallel
and orthogonal parts sum back to the original guidance vector, and with the
other two disabled nothing else touches it.

**This is not a bitwise identity, and the neutral setting is not a usable A/B
baseline.** The projection decomposes the guidance vector in double precision
and casts the parts back, so the round trip perturbs the low-order bits. A
CPU numerical suite confirms the identity to a tolerance of `0.000001`, but
during a real sampling run the solver amplifies that perturbation. Measured
on SDXL with `kutta4`, Align Your Steps, 35 steps, CFG 7 and a fixed seed,
the neutral setting differs from a bare run by a mean absolute RGB difference
of about `0.74`, with 71 % of pixels differing.

To measure what APG actually changes, **disable the extension** rather than
neutralising it.

### Suggested starting points

| Situation | Eta | Norm Threshold | Momentum |
|---|---|---|---|
| Paper default (single extension, high CFG) | 0.0 | 15.0 | 0.0 |
| Result looks washed out or low-contrast | 0.3 – 0.7 | 15.0 | 0.0 |
| **Stacking with other guidance extensions** | 1.0 | 0.0 | 0.0 |

When APG runs on top of TCFG / SkimmedCFG / DifferenceCFG and others, the
paper defaults are usually too strong: each extension in the chain is already
reducing the guidance magnitude. Start from the neutral settings and lower
Eta gradually from there.

---

## ⚠ Momentum and high-order solvers

**Momentum is not recommended with multi-stage or adaptive ODE solvers.**

The momentum buffer keeps a running average **across model evaluations**, so
the integrand is no longer a function of `(x, sigma)` alone. This breaks the
stateless right-hand-side assumption that ODE solvers rely on:

- Multi-stage methods evaluate the model several times per step
  (`kutta4` = 4 evaluations), so the buffer accumulates several times faster
  than it does with a single-stage method at the same step count.
- Adaptive step control re-evaluates rejected steps, making the accumulation
  rate depend on the tolerance settings.

Fixed-seed measurements (SDXL, 35 steps, Align Your Steps, CFG 7) showed that
this is not simply a matter of scaling the coefficient. With a 4-stage solver,
changing a momentum-related parameter by a very small amount moved the result
about as far as it was from the baseline in the first place — the output
jumped to an unrelated solution instead of changing proportionally. With a
single-stage solver the same change produced roughly half the displacement,
but the value-to-result relationship was still not monotonic.

**Interpretation.** Once momentum is enabled, a high-order solver's
intermediate stages no longer improve the estimate; they amplify small early
perturbations instead. The higher the order, the stronger the amplification —
so the accuracy you are paying extra model evaluations for is lost.

**Recommendation.**

| Sampler | Momentum |
|---|---|
| Euler, LMS, and other single-stage methods | Usable |
| Heun, DPM++ 2M / 3M, and other multi-stage methods | Not recommended |
| TDE Sampler / RK Sampler (multi-stage or adaptive solvers) | Not recommended |

Momentum defaults to `0` (off), which keeps APG a purely stateless
per-evaluation transform and safe to combine with any sampler.

---

## Composition with other extensions

Execution order in the chain is decided by priority insertion
(`_priority_insert_pre_cfg` / `_priority_insert_post_cfg` in `core.py`),
not by `sorting_priority`, which only controls where this extension's
accordion is drawn in the UI. APG registers at priority 14.5:

```
TCFG (13.0) → SkimmedCFG (14.0) → DifferenceCFG (14.2) → APG (14.5)
    → CFG → CFGZeroStar (15.0) → FreSca (15.2) → MaHiRo (15.5)
    → CFGNorm (16.0) → CFGRegulator (16.5)
```

This matches the "final polish before CFG" role recommended for APG.

On Forge Neo, TCFG's damped uncond is read from the shared
`model_options` dict when TCFG ran earlier in the same post-CFG call.

Set `SD_WEBUI_SETI_DEBUG=1` before launching to have the assembled chain
printed at sampling time:

```
[APG] pre-CFG chain: _tcfg_pre_cfg_fn(13.0) -> ... -> _apg_pre_cfg_fn(14.5)
```

If the printed order differs from the list above, something in the chain
is not participating in priority insertion.

---

## Differences from the ComfyUI built-in node

Both differences are deliberate.

| | This extension | ComfyUI `APG` node |
|---|---|---|
| Final combination | `cond + (cond_scale - 1) * update` (paper Algorithm 1) | effectively `cond + cond_scale * update` |
| Reduction dims | all non-batch dims (`range(1, ndim)`) | fixed `dim=[-1, -2, -3]` |
| Precision of the projection | computed in `double`, cast back (paper Algorithm 1) | computed in the input dtype |
| Momentum reset | none; a fresh buffer per sampling pass | buffer cleared whenever sigma increases |

The first means the neutral settings reduce to standard CFG algebraically,
which the ComfyUI node cannot do at any setting — it is always one guidance
unit stronger. See [Neutral settings](#neutral-settings) for why the
algebraic identity does not carry through to a bitwise one.

The second is identical to the paper for 4-D `(B, C, H, W)` latents but keeps
the paper's per-sample semantics for 5-D `(B, C, T, H, W)` latents instead of
silently becoming per-channel. HuggingFace diffusers'
`AdaptiveProjectedGuidance` makes the same choice (`norm_dim=None`).

The fourth is a removal. The script layer builds a fresh closure, and
therefore a fresh momentum buffer, for every sampling pass, so a running
average can never survive into an unrelated run. A sigma-increase guard adds
nothing on top of that, and it actively misfires on adaptive-step solvers
such as those in sd-webui-TDE-Sampler and sd-webui-RK-Sampler, which
legitimately re-try a rejected step at a larger sigma. Clearing the buffer
there discards a valid running average mid-trajectory.

---

## Infotext keys

```
APG Eta, APG Norm Threshold, APG Momentum
```

`APG Eta` is written only while APG is active, so its presence in the
infotext doubles as the enable marker on read-back.

> Images generated with v1.x may also carry an `APG Adaptive Momentum` key.
> That feature has been removed; the key is ignored on read and does not
> prevent the rest of the settings from being restored.

---

## Removed feature: Adaptive Momentum

Releases before v2.0 offered an **Adaptive Momentum** slider, an original
addition that faded the momentum coefficient to zero over the early part of
the sigma schedule. It has been removed.

Fixed-seed testing showed that the parameter did not behave as a continuous
control at any setting:

- Below roughly `0.21` it had no effect at all (bit-identical to momentum
  applied normally).
- Above that threshold it took effect, but the value and the result were not
  related monotonically — neighbouring values produced results as far apart
  from each other as they were from the baseline.
- Raising it toward `1.0` did not converge on momentum-off behaviour, because
  the coefficient decays but never stops being applied.

The exact threshold also moved depending on the solver, since it is the
number of model evaluations before the cut-off that decides the outcome.
A slider whose value cannot be reasoned about is worse than no slider, so it
was dropped rather than documented as a quirk.

---

<a id="日本語"></a>

# sd-webui-APG（日本語）

Stable Diffusion WebUI（Forge 系）向けの Adaptive Projected Guidance（APG）拡張機能です。

[**Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models**](https://arxiv.org/abs/2410.02416)（ICLR 2025）の Algorithm 1 の実装です。

> APG は高い CFG スケールで発生する彩度過多やアーチファクトを抑えます。
> 色飛びやコントラストの破綻を気にせず CFG を上げ、プロンプトへの追従を
> 強められます。

---

## 特徴

- ComfyUI ノードではなく**論文の式**に忠実。中立設定で標準 CFG に**代数的に**帰着します（ビット単位の恒等ではありません。詳細は[中立設定](#中立設定)を参照してください）。
- **reForge / Forge Classic**（Pre-CFG フック）と **Forge Neo**（Post-CFG フック）に対応。バックエンドは自動判別します。
- 階数非依存の射影により、4 次元の SDXL latent と 5 次元の Anima / NextDiT latent の両方に対応。
- `_sd_webui_priority = 14.5` による優先度挿入で、他のガイダンス拡張と正しい順序で合成されます。詳細は[他の拡張機能との合成](#他の拡張機能との合成)を参照してください。
- 全パラメータの XYZ Grid 軸を提供。
- 生成パラメータを PNG infotext に埋め込み、再現可能です。

---

## インストール

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-APG
```

> この拡張機能は Forge バックエンドのフック API を利用します。
> A1111（AUTOMATIC1111）では動作しません。

---

## 動作原理

APG はガイダンスベクトルを、条件付き予測に**平行な成分**と**直交する成分**に分解します。平行成分は latent を既に進んでいる方向へさらに押し込むもので、高 CFG における彩度過多の主因です。これを縮小すれば、意味的な誘導（直交成分）を保ったまま色飛びだけを取り除けます。

```
diff      = cond - uncond
diff      = momentum_buffer.update(diff)      （任意、beta != 0 のとき）
diff      = L2 ノルムを norm_threshold にクランプ  （任意、threshold > 0 のとき）
par, orth = diff を cond に射影して分解
update    = orth + eta * par
final     = cond + (cond_scale - 1) * update
```

---

## パラメータ

| 項目 | 範囲 | 既定値 | 説明 |
|---|---|---|---|
| **Enable APG** | — | オフ | 有効化スイッチ。 |
| **Eta** | 0.0 〜 2.0 | 0.0 | 平行成分をどれだけ残すか。`0` が論文推奨の既定値、`1.0` で全て残す（＝射影が無効）。結果が平坦・低彩度に見える場合は上げます。 |
| **Norm Threshold** | 0.0 〜 50.0 | 15.0 | ガイダンスベクトルのサンプルごとの L2 クランプ。`0` で無効。論文は SDXL に `15` を使用。 |
| **Momentum** | -1.5 〜 1.0 | 0.0 | 移動平均係数（beta）。`0` で無効。論文は `-0.5` などの負値を使用。**有効化する前に下の警告を必ずお読みください。** |

### 中立設定

```
Eta 1.0 / Norm Threshold 0 / Momentum 0
```

この設定は標準 CFG に**代数的に**帰着します。`Eta 1.0` では平行成分と直交成分の和が元のガイダンスベクトルに戻り、他の 2 つを無効にすれば他に手を加えるものがないためです。

**これはビット単位の恒等ではなく、中立設定は A/B 基準として使えません。** 射影はガイダンスベクトルを倍精度で分解して各成分を戻すため、往復の丸めが下位ビットを変化させます。CPU 数値スイートでは許容誤差 `0.000001` で恒等性が確認されていますが、実際のサンプリングではソルバーがこの摂動を増幅します。SDXL / `kutta4` / Align Your Steps / 35 ステップ / CFG 7 / 固定シードで実測したところ、中立設定と素の実行との平均絶対 RGB 差は約 `0.74`、差分画素は 71 % でした。

APG が実際に何を変えているかを測るには、中立設定にするのではなく**拡張機能自体を無効化**してください。

### 設定の目安

| 状況 | Eta | Norm Threshold | Momentum |
|---|---|---|---|
| 論文既定値（単独使用・高 CFG） | 0.0 | 15.0 | 0.0 |
| 結果が眠い・コントラスト不足 | 0.3 〜 0.7 | 15.0 | 0.0 |
| **他のガイダンス拡張と併用** | 1.0 | 0.0 | 0.0 |

TCFG / SkimmedCFG / DifferenceCFG などの上に APG を重ねる場合、論文の既定値は通常強すぎます。チェーン内の各拡張がすでにガイダンスの大きさを削っているためです。中立設定から始め、Eta を少しずつ下げていくことをお勧めします。

---

## ⚠ Momentum と高次ソルバーについて

**Momentum は多段ソルバーおよび可変ステップソルバーとの併用を推奨しません。**

momentum バッファは**モデル評価をまたいで**移動平均を保持します。そのため被積分関数が `(x, sigma)` だけの関数ではなくなり、ODE ソルバーが前提とする「右辺が無状態である」という条件が崩れます。

- 多段法は 1 ステップあたり複数回モデルを評価するため（`kutta4` は 4 回）、同じステップ数でも単段法よりバッファの蓄積が数倍速くなります。
- 可変ステップ制御は棄却されたステップを再評価するため、蓄積速度が許容誤差の設定に依存します。

固定シードでの実測（SDXL / 35 ステップ / Align Your Steps / CFG 7）では、これが単なる係数のスケールの違いではないことが確認されました。4 段ソルバーでは、momentum 関連のパラメータをごくわずかに変えただけで、結果が「元の基準からの距離とほぼ同じだけ」動きました。つまり比例して変化するのではなく、**無関係な別の解に飛んだ**ということです。単段ソルバーでは変位はおよそ半分でしたが、値と結果の対応が単調でない点は変わりませんでした。

**解釈。** momentum を有効にした時点で、高次ソルバーの中間段はもはや推定精度を改善しません。代わりに初期の微小な摂動を増幅します。次数が高いほど増幅は強くなるため、**余分なモデル評価を払って得ていたはずの精度が失われます**。

**推奨。**

| サンプラー | Momentum |
|---|---|
| Euler、LMS その他の単段法 | 使用可 |
| Heun、DPM++ 2M / 3M その他の多段法 | 非推奨 |
| TDE Sampler / RK Sampler（多段・可変ステップソルバー） | 非推奨 |

Momentum の既定値は `0`（オフ）です。この状態では APG は完全に無状態な評価ごとの変換であり、どのサンプラーと組み合わせても安全です。

---

## 他の拡張機能との合成

チェーン内の実行順序は、優先度挿入（`core.py` の `_priority_insert_pre_cfg` /
`_priority_insert_post_cfg`）によって決まります。`sorting_priority` ではありません。
`sorting_priority` は本拡張機能のアコーディオンが UI 上のどこに描画されるかのみを制御します。
APG は優先度 14.5 で登録されます。

```
TCFG (13.0) → SkimmedCFG (14.0) → DifferenceCFG (14.2) → APG (14.5)
    → CFG → CFGZeroStar (15.0) → FreSca (15.2) → MaHiRo (15.5)
    → CFGNorm (16.0) → CFGRegulator (16.5)
```

これは APG に推奨される「CFG 直前の最終調整」という役割に一致します。

起動前に `SD_WEBUI_SETI_DEBUG=1` を設定すると、サンプリング時に組み立てられた
チェーンが出力されます。

```
[APG] pre-CFG chain: _tcfg_pre_cfg_fn(13.0) -> ... -> _apg_pre_cfg_fn(14.5)
```

出力された順序が上記と異なる場合、チェーン内のいずれかが優先度挿入に参加していません。

Forge Neo では、同一の post-CFG 呼び出し内で TCFG が先に実行されていた場合、共有の `model_options` から TCFG の減衰済み uncond を読み取ります。

---

## ComfyUI 組み込みノードとの相違点

いずれの相違点も意図的なものです。

| | 本拡張機能 | ComfyUI `APG` ノード |
|---|---|---|
| 最終合成 | `cond + (cond_scale - 1) * update`（論文 Algorithm 1） | 実質的に `cond + cond_scale * update` |
| 縮約次元 | バッチ以外の全次元（`range(1, ndim)`） | 固定の `dim=[-1, -2, -3]` |
| 射影の精度 | `double` で計算して戻す（論文 Algorithm 1） | 入力の dtype のまま計算 |
| Momentum リセット | なし。サンプリングパスごとに新しいバッファ | σ が増加するたびにバッファを破棄 |

第一の相違により、中立設定が標準 CFG に代数的に帰着します。ComfyUI ノードはどの設定でもこれができず、常にガイダンス 1 単位分強くなります。代数的な恒等がビット単位の一致にならない理由については「中立設定」の節を参照してください。

第二の相違は 4 次元 `(B, C, H, W)` の潜在表現では論文と同一です。5 次元 `(B, C, T, H, W)` の潜在表現において、暗黙にチャネル単位へ変わってしまうことを避け、論文のサンプル単位の意味を保ちます。HuggingFace diffusers の `AdaptiveProjectedGuidance` も同じ選択をしています（`norm_dim=None`）。

第四の相違は削除です。スクリプト層がサンプリングパスごとに新しいクロージャ、したがって新しい momentum バッファを生成するため、移動平均が無関係な実行へ持ち越されることはありません。σ 増加ガードはその上に何も付け加えず、むしろ sd-webui-TDE-Sampler や sd-webui-RK-Sampler のような適応ステップソルバーで誤発火します。これらは棄却したステップをより大きい σ で再試行する正当な動作をするため、そこでバッファを破棄すると軌道の途中で有効な移動平均を捨てることになります。

---

## infotext キー

```
APG Eta, APG Norm Threshold, APG Momentum
```

`APG Eta` は APG が有効なときのみ書き込まれるため、infotext 中の存在自体が読み込み時の有効化マーカーを兼ねます。

> v1.x で生成した画像には `APG Adaptive Momentum` キーが含まれている場合があります。この機能は削除されましたが、当該キーは読み込み時に無視されるだけで、他の設定の復元を妨げません。

---

## 削除された機能: Adaptive Momentum

v2.0 より前のリリースには **Adaptive Momentum** スライダーがありました。これは本移植独自の追加機能で、シグマスケジュールの前半にかけて momentum 係数をゼロへ減衰させるものでしたが、削除されました。

固定シードでの検証により、この値がどの設定でも連続的な制御として機能しないことが判明したためです。

- おおよそ `0.21` 以下では効果が一切ありませんでした（momentum をそのまま適用した場合とビット単位で一致）。
- それ以上では効果が出ますが、値と結果が単調に対応しませんでした。隣り合う値どうしの結果が、基準からの距離と同じだけ離れていました。
- `1.0` に近づけても momentum オフの挙動には収束しません。係数は減衰しますが、適用され続けること自体は止まらないためです。

閾値の位置もソルバーによって移動します。結果を決めているのが打ち切りまでのモデル評価回数だからです。値について推論できないスライダーは、無いほうがましだと判断し、仕様として文書化するのではなく削除しました。

---

## License / ライセンス

**MIT License** — see [LICENSE](LICENSE).

Copyright (c) 2026 seti9585

## Attribution / 典拠

**Paper / 論文**

Sadat, S., Hilliges, O., & Weber, R. M.
*Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models.*
ICLR 2025. [arXiv:2410.02416](https://arxiv.org/abs/2410.02416)

The algorithm in this extension is written from Algorithm 1 of that paper,
which the authors publish as the reference implementation of APG. The
`MomentumBuffer` class, the double-precision projection, the scalar-zero
initial running average and the `(guidance_scale - 1)` final combination all
follow that listing.

本拡張機能のアルゴリズムは、上記論文の Algorithm 1 をもとに記述しています。同 Algorithm は著者らが APG の参照実装として公開しているものです。`MomentumBuffer` クラス、倍精度での射影、移動平均のスカラー 0 初期化、`(guidance_scale - 1)` による最終合成は、いずれも同 Algorithm に従っています。

## Acknowledgements / 謝辞

**Shiba-2-shiba**

The author first learned of APG through the note.com articles of
[**Shiba-2-shiba**](https://note.com/gentle_murre488), whose
[TCFG-APG-Mahiro-for-ForgeClassic](https://github.com/Shiba-2-shiba/TCFG-APG-Mahiro-for-ForgeClassic)
implementation for Forge Classic was also consulted. Development of this
whole extension suite started from that work. The pointer that made APG
knowable is gratefully acknowledged.

APG の存在は [**Shiba-2-shiba**](https://note.com/gentle_murre488) 氏の note.com の記事によって知りました。Forge Classic 向けの実装である [TCFG-APG-Mahiro-for-ForgeClassic](https://github.com/Shiba-2-shiba/TCFG-APG-Mahiro-for-ForgeClassic) も参考にさせていただいています。本拡張スイート全体の開発は、この記事と実装をきっかけに始まりました。APG を知るきっかけを与えてくださったことに深く感謝します。

**ComfyUI**

The built-in `APG` node of [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
(`comfy_extras/nodes_apg.py`, GPL-3.0) was read throughout development as a
working reference for how APG is wired into a pre-CFG hook. Where this
extension departs from it, the departures are listed above and are
deliberate. No code was carried over: the algorithm follows the paper
listing, and the one behaviour that had been aligned with the node — clearing
the momentum buffer on a sigma increase — has been removed. Thanks are due to
comfyanonymous and the ComfyUI contributors regardless.

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) の組み込み `APG` ノード（`comfy_extras/nodes_apg.py`、GPL-3.0）は、APG を pre-CFG フックへ組み込む実際の方法を示す参考として、開発を通じて参照しました。本拡張機能が異なる点は上記のとおりで、いずれも意図的なものです。コードの流用はありません。アルゴリズムは論文の Algorithm に従っており、唯一同ノードに挙動を合わせていた「σ 増加時の momentum バッファ破棄」も削除しました。それでもなお、comfyanonymous 氏および ComfyUI コントリビューターの皆様に感謝します。

**Reference implementations / 参考実装**

- ComfyUI built-in `APG` node (`comfy_extras/nodes_apg.py`)
- HuggingFace diffusers `AdaptiveProjectedGuidance`
