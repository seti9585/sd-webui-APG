# sd-webui-APG

**EN** | [日本語](#日本語)

Adaptive Projected Guidance (APG) for Forge-based Stable Diffusion WebUIs.

APG keeps colors from burning out when CFG is high.

Raising CFG makes the image follow the prompt more closely, but it also pushes colors and contrast too far: flat areas turn into solid, fully saturated color and highlights blow out. APG splits the push that CFG adds into two parts and weakens only the part that causes this burn.

```text
CFG push = part that moves the image toward the prompt   -> kept
         + part that just makes the image "more of itself" -> weakened by APG
                                                              (source of the burn)
```

The result is that you can use a higher CFG, or keep your usual CFG on a model that tends to oversaturate, without the colors breaking down.

APG is based on *Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models* (Sadat et al., ICLR 2025), Algorithm 1.

## Measured result

Common conditions: reForge / SDXL (amanatsuIllustrious v11, an Illustrious-based model that oversaturates easily) / TDE Sampler (kutta4) / Align Your Steps / 35 steps / 896x1152 / one fixed seed / Norm Threshold 15 / Momentum 0.

Saturation is the average HSV saturation (0 = grey, 255 = fully saturated). "Clipped" is the share of pixels whose saturation hits the upper limit.

```text
                       CFG 7                  CFG 15
                   saturation  clipped    saturation  clipped
APG off              153.9      5.6 %       159.3     12.2 %
APG Eta 1.0          134.1      0.0 %       144.9      0.2 %
APG Eta 0.5          117.8      0.0 %       132.3      0.0 %
APG Eta 0.0           91.1      0.0 %        85.3      0.0 %
```

The main change is how far the colors are pushed; fine detail is kept. Small objects and background details can still change, as with any guidance change. Higher Eta keeps more color. These figures come from one model and one seed, so treat them as a direction rather than exact values for your setup.

---

## Installation

**Extensions -> Install from URL**

```text
https://github.com/seti9585/sd-webui-APG
```

Restart the WebUI after installation. When updating from an older version, **restart the WebUI** and reload the browser page (Ctrl + F5) instead of using Reload UI.

> This extension uses the Forge backend hook API. It does not work on A1111 (AUTOMATIC1111).

---

## Quick start

1. Open the **APG (Adaptive Projected Guidance)** panel.
2. Enable **Enable APG**.
3. Leave the other settings at their defaults (Eta 0, Norm Threshold 15, Momentum 0).
4. Generate normally.

The defaults are the paper's values and suppress color strongly. If the image looks too grey or flat, raise **Eta** first.

---

## Tuning

**Colors look washed out or flat -> raise Eta (0.3 to 1.0)**  
**Colors still burn -> lower Eta, or lower Norm Threshold slightly**  
**Image looks soft or sleepy -> raise Norm Threshold (too low, such as 2 to 4, weakens the prompt)**  
**Want a slightly calmer result -> Momentum -0.1 to -0.5 (see below)**

---

## Parameters

| Parameter | Range | Default | Description |
| --- | ---: | ---: | --- |
| **Enable APG** | - | Off | Master switch. |
| **Eta** | 0.0 - 2.0 | 0.0 | How much of the "burn" part to keep. `0` removes it (strongest effect, paper default), `1.0` keeps it fully. Raising it brings back color and contrast step by step. |
| **Norm Threshold** | 0.0 - 50.0 | 15.0 | Upper limit on the strength of the CFG push. `0` disables the limit. Lower values hold the push down harder. The paper uses `15` for SDXL. |
| **Momentum** | -1.5 - 1.0 | 0.0 | Mixes the current guidance with a weighted history of previous model evaluations. `0` disables it. Negative values make the result slightly calmer. |

### Norm Threshold and image size

The strength of the push is measured over the whole latent, so it grows with image size. The same Norm Threshold holds the push down harder on a larger image (for example, the Hires.fix pass). If the Hires.fix result looks weaker than the first pass, raise Norm Threshold for that pass.

### Momentum and samplers

Momentum works as a smooth control between `0` and about `-0.5`: the more negative, the calmer the result.

- Its strength depends on the sampler. At `-0.15`, Euler changed the image about 1.8 times as much as kutta4 (4 model evaluations per step). Re-tune it when you change the sampler.
- `-1.0` or below is unstable and the image collapses into a flat grey result. Do not go that far.
- Adaptive-step samplers (such as DPM adaptive) re-evaluate rejected steps, so the strength also depends on their tolerance settings.

### Comparing with and without APG

To see what APG changes, **turn Enable APG off** rather than setting Eta 1.0 / Norm Threshold 0 / Momentum 0. Those "neutral" values equal plain CFG mathematically, but not to the last bit, so the image still changes slightly.

---

## Using APG with other guidance extensions

Other extensions that change the CFG push can be combined with APG. The ones that also reduce color (for example [TCFG](https://github.com/seti9585/sd-webui-TCFG)) add to APG's effect, so raise Eta when you stack them. Extensions that push the image toward the positive prompt (for example [MaHiRo](https://github.com/seti9585/sd-webui-MaHiRo)) bring saturation back, so lower Eta.

Measured at CFG 15 with the same conditions as above:

```text
                                   saturation  clipped
MaHiRo only                          171.3     33.5 %
APG Eta 0.0 + MaHiRo                 148.5      1.1 %
APG Eta 0.5 + MaHiRo                 160.4     16.4 %
TCFG + APG Eta 0.5 + MaHiRo          142.9      0.1 %
```

Both `APG Eta 0 + MaHiRo` and `TCFG + APG Eta 0.5 + MaHiRo` gave a bright image with very little saturation clipping at CFG 15. Other guidance extensions should be tested individually.

### Execution order

APG runs after CFG is combined (post-CFG), at priority 15.4, on every backend. When the extensions below are enabled, they run in this order regardless of which panel you touch first. Where TCFG / SkimmedCFG / DifferenceCFG run depends on the backend:

```text
reForge / Forge Classic:
TCFG (13.0) -> SkimmedCFG (14.0) -> DifferenceCFG (14.2) -> CFG
  -> CFGZeroStar (15.0) -> FreSca (15.2) -> APG (15.4)
  -> MaHiRo (15.5) -> CFGNorm (16.0) -> CFGRegulator (16.5)

Forge Neo:
CFG
  -> TCFG (13.0) -> SkimmedCFG (14.0) -> DifferenceCFG (14.2)
  -> CFGZeroStar (15.0) -> FreSca (15.2) -> APG (15.4)
  -> MaHiRo (15.5) -> CFGNorm (16.0) -> CFGRegulator (16.5)
```

On Forge Neo every hook, including TCFG / SkimmedCFG / DifferenceCFG, runs after CFG is combined; the priority numbers are the same on both backends.

APG reads the result of the extensions before it and does not discard it. Extensions by other authors are not reordered; where they land relative to APG depends on the extension load order.

---

## Compatibility

| Target | Status |
| --- | --- |
| reForge | Supported (measurements above were taken here) |
| Stable Diffusion WebUI Forge / Forge Classic | Supported design target |
| Forge Neo | Supported design target |
| SDXL-family models | Primary target |
| Anima / NextDiT | Architecture-compatible / validation pending |
| A1111 | Not supported |

XYZ Grid axes are provided for Enabled, Eta, Norm Threshold and Momentum. Settings are saved in the PNG infotext (`APG Eta`, `APG Norm Threshold`, `APG Momentum`) and restored with PNG Info -> Send to txt2img.

---

## Changes in v3.0

- APG now runs after CFG is combined on every backend, after FreSca and before MaHiRo. In v2.x it ran before CFG on reForge.
- **Images differ from v2.x even with the same seed and settings.** In a fixed-seed comparison (CFG 15, TCFG + APG Eta 0.5 + MaHiRo) the saturation level was the same in both versions, while FreSca's settings now carry through to the result as intended.
- On Forge Neo, APG no longer overwrites the result produced by the preceding SkimmedCFG or DifferenceCFG hook. The existing composition limitation between SkimmedCFG and DifferenceCFG themselves is unchanged.
- XYZ Grid cells with APG disabled are no longer saved with APG keys in their infotext.
- The Momentum notes were rewritten from new measurements. Earlier releases called Momentum unpredictable with multi-stage samplers; that measurement was taken with the removed Adaptive Momentum option (removed in v2.0) and does not apply to Momentum itself.

---

## Debug output

Set the environment variable before launching the WebUI:

```powershell
$env:SD_WEBUI_SETI_DEBUG = "1"
```

The console then shows the registered hook and, once per generation, the actual post-CFG order:

```text
[APG] registered post-CFG hook (reForge / Forge Classic backend), priority=15.4
[APG] post-CFG chain: fresca_hook(15.2) -> _apg_post_cfg_fn(15.4) -> _mahiro_fn(15.5)
```

---

## How it works

APG takes the CFG push that the extensions before it produced, measured against the conditional (positive) prediction, and applies the paper's Algorithm 1:

```text
G         = denoised - cond                (push produced so far)
diff      = G / (cond_scale - 1)           (back to the paper's cond - uncond scale)
diff      = momentum_buffer.update(diff)   (Momentum != 0 only)
diff      = clamp || diff ||_2 to norm_threshold   (Norm Threshold > 0 only)
par, orth = split diff into the parts parallel / orthogonal to cond
update    = orth + eta * par
out       = cond + (cond_scale - 1) * update
```

The parallel part points the same way as the image already does, which is why amplifying it with CFG burns the colors. When no other post-CFG extension runs before APG, `diff` is exactly `cond - uncond` and the result is the paper's formula.

### Differences from the ComfyUI built-in node

| | This extension | ComfyUI `APG` node |
| --- | --- | --- |
| Final combination | `cond + (cond_scale - 1) * update` (paper) | effectively `cond + cond_scale * update` |
| Reduction dims | all non-batch dims | fixed `dim=[-1, -2, -3]` |
| Projection precision | double, cast back (paper) | input dtype |
| Momentum reset | fresh buffer per sampling pass | cleared whenever sigma increases |

The first difference means the neutral settings equal plain CFG, which the ComfyUI node cannot do at any setting. The second keeps the paper's per-image behaviour for 5-D Anima / NextDiT latents, the same choice HuggingFace diffusers makes. The sigma-increase reset is not used because it misfires on adaptive-step samplers, which legitimately retry a rejected step at a larger sigma.

---

<a id="日本語"></a>

# 日本語

**[English](#sd-webui-apg)** | 日本語

Forge 系 Stable Diffusion WebUI 向けの Adaptive Projected Guidance（APG）拡張機能です。

APG は、**CFG を上げたときに色が焼けるのを防ぎます。**

CFG を上げるとプロンプトへの追従は強くなりますが、色とコントラストも押し出されすぎます。平らな部分がべったりした原色になり、明るい部分は白く飛びます。APG は CFG が加える「押し」を 2 つに分け、焼けの原因になる側だけを弱めます。

```text
CFG の押し = 画像をプロンプトへ近づける成分        -> そのまま残す
           + 画像を「今の自分」へさらに強める成分  -> APG が弱める
                                                    （焼けの原因）
```

その結果、高い CFG を使っても、あるいは過飽和しやすいモデルでいつもの CFG を使っても、色が破綻しにくくなります。

APG は論文 *Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models*（Sadat ほか、ICLR 2025）の Algorithm 1 に基づいています。

## 実測結果

共通条件: reForge / SDXL（amanatsuIllustrious v11。過飽和しやすい Illustrious 系モデル）/ TDE Sampler（kutta4）/ Align Your Steps / 35 steps / 896x1152 / 固定 seed 1 つ / Norm Threshold 15 / Momentum 0。

彩度は HSV の彩度の平均です（0 が灰色、255 が最も鮮やか）。「張り付き」は、彩度が上限に達した画素の割合です。

```text
                        CFG 7                CFG 15
                    彩度    張り付き     彩度    張り付き
APG OFF            153.9     5.6 %     159.3    12.2 %
APG Eta 1.0        134.1     0.0 %     144.9     0.2 %
APG Eta 0.5        117.8     0.0 %     132.3     0.0 %
APG Eta 0.0         91.1     0.0 %      85.3     0.0 %
```

主に変わるのは色の押し出され方で、細部は保たれます。ただし、ガイダンスを変える以上、小物や背景の細部が変わることはあります。Eta を上げるほど色が残ります。1 つのモデル・1 つの seed での結果なので、お使いの環境での正確な値ではなく、傾向として見てください。

---

## インストール

**Extensions -> Install from URL**

```text
https://github.com/seti9585/sd-webui-APG
```

インストール後、WebUI を再起動してください。旧版から更新した場合も、Reload UI ではなく **WebUI を再起動**し、ブラウザのページを再読み込み（Ctrl + F5）してください。

> この拡張機能は Forge バックエンドのフック API を使います。A1111（AUTOMATIC1111）では動作しません。

---

## まず使う

1. **APG (Adaptive Projected Guidance)** パネルを開く
2. **Enable APG** を ON
3. 他は既定値のまま（Eta 0、Norm Threshold 15、Momentum 0）
4. そのまま生成

既定値は論文の値で、色をかなり強く抑えます。灰色っぽい・平坦に見える場合は、まず **Eta** を上げてください。

---

## 調整の目安

**色が抜ける・平坦に見える -> Eta を上げる（0.3 から 1.0）**  
**まだ色が焼ける -> Eta を下げる、または Norm Threshold を少し下げる**  
**眠い・ぼやけた絵になる -> Norm Threshold を上げる（2 から 4 のように低すぎるとプロンプトの効きが弱まる）**  
**少し落ち着いた絵にしたい -> Momentum を -0.1 から -0.5（後述）**

---

## パラメータ

| パラメータ | 範囲 | 既定値 | 説明 |
| --- | ---: | ---: | --- |
| **Enable APG** | - | オフ | 有効化スイッチ。 |
| **Eta** | 0.0 - 2.0 | 0.0 | 「焼け」の成分をどれだけ残すか。`0` で取り除く（効果最大、論文の既定値）、`1.0` ですべて残す。上げるほど色とコントラストが段階的に戻ります。 |
| **Norm Threshold** | 0.0 - 50.0 | 15.0 | CFG の押しの強さの上限。`0` で上限なし。小さいほど強く抑えます。論文は SDXL に `15` を使用。 |
| **Momentum** | -1.5 - 1.0 | 0.0 | 現在のガイダンスに、過去のモデル評価の履歴を重み付きで加えます。`0` で無効。負の値ほど少し落ち着いた絵になります。 |

### Norm Threshold と画像サイズ

押しの強さは潜在空間全体で測るため、画像が大きいほど値も大きくなります。同じ Norm Threshold でも、大きな画像（Hires.fix の 2 パス目など）ほど強く抑えることになります。Hires.fix の結果が 1 パス目より弱く見える場合は、Norm Threshold を上げてください。

### Momentum とサンプラー

Momentum は `0` から `-0.5` 程度の範囲で、なめらかに効きます。負の値ほど落ち着いた絵になります。

- 効きの強さはサンプラーで変わります。`-0.15` で、Euler は kutta4（1 ステップにモデルを 4 回評価）の約 1.8 倍、画像が変わりました。サンプラーを変えたら値を見直してください。
- `-1.0` 以下は不安定で、灰色の平坦な絵に崩れます。そこまで下げないでください。
- 可変ステップのサンプラー（DPM adaptive など）は、棄却したステップを評価し直すため、効きの強さが許容誤差の設定にも左右されます。

### APG の効果を比べるとき

APG が何を変えているかを見るには、Eta 1.0 / Norm Threshold 0 / Momentum 0 にするのではなく、**Enable APG をオフ**にしてください。この「中立」の値は数式の上では通常の CFG と同じですが、計算の最後の桁まで一致するわけではないので、画像はわずかに変わります。

---

## 他のガイダンス拡張と組み合わせる

CFG の押しを変える他の拡張機能と組み合わせられます。色を抑える方向の拡張（例: [TCFG](https://github.com/seti9585/sd-webui-TCFG)）は APG の効果に上乗せされるので、重ねるときは Eta を上げてください。画像をポジティブプロンプト側へ押し出す拡張（例: [MaHiRo](https://github.com/seti9585/sd-webui-MaHiRo)）は彩度を戻すので、Eta を下げてください。

上と同じ条件、CFG 15 での実測:

```text
                                    彩度    張り付き
MaHiRo のみ                        171.3    33.5 %
APG Eta 0.0 + MaHiRo               148.5     1.1 %
APG Eta 0.5 + MaHiRo               160.4    16.4 %
TCFG + APG Eta 0.5 + MaHiRo        142.9     0.1 %
```

`APG Eta 0 + MaHiRo` と `TCFG + APG Eta 0.5 + MaHiRo` のどちらも、CFG 15 で張り付きを大幅に抑えながら、鮮やかな絵になりました。その他のガイダンス拡張は、個別に試してください。

### 実行順

APG は、すべてのバックエンドで CFG の合成後（post-CFG）に、優先度 15.4 で動きます。下の拡張機能が有効なときは、パネルを操作した順番に関係なく、次の順に実行されます。TCFG / SkimmedCFG / DifferenceCFG が動く位置は、バックエンドによって違います。

```text
reForge / Forge Classic:
TCFG (13.0) -> SkimmedCFG (14.0) -> DifferenceCFG (14.2) -> CFG
  -> CFGZeroStar (15.0) -> FreSca (15.2) -> APG (15.4)
  -> MaHiRo (15.5) -> CFGNorm (16.0) -> CFGRegulator (16.5)

Forge Neo:
CFG
  -> TCFG (13.0) -> SkimmedCFG (14.0) -> DifferenceCFG (14.2)
  -> CFGZeroStar (15.0) -> FreSca (15.2) -> APG (15.4)
  -> MaHiRo (15.5) -> CFGNorm (16.0) -> CFGRegulator (16.5)
```

Forge Neo では、TCFG / SkimmedCFG / DifferenceCFG を含むすべてのフックが CFG の合成後に動きます。優先度の番号は両方のバックエンドで同じです。

APG は前に動いた拡張機能の結果を受け取り、捨てずに使います。他の作者の拡張機能の順番は変更しません。それらと APG の前後関係は、拡張機能の読み込み順で決まります。

---

## 対応状況

| 対象 | 状況 |
| --- | --- |
| reForge | 対応（上の実測はここで取得） |
| Stable Diffusion WebUI Forge / Forge Classic | 対応設計 |
| Forge Neo | 対応設計 |
| SDXL 系モデル | 主対象 |
| Anima / NextDiT | 構造上対応 / 検証待ち |
| A1111 | 非対応 |

XYZ Grid に Enabled、Eta、Norm Threshold、Momentum の軸を追加します。設定は PNG infotext（`APG Eta`、`APG Norm Threshold`、`APG Momentum`）に保存され、PNG Info -> Send to txt2img で復元できます。

---

## v3.0 での変更点

- すべてのバックエンドで、APG は CFG の合成後に、FreSca の後・MaHiRo の前で動くようになりました。v2.x までは reForge では CFG の前で動いていました。
- **同じ seed・同じ設定でも、v2.x とは画像が変わります。** 固定 seed での比較（CFG 15、TCFG + APG Eta 0.5 + MaHiRo）では、彩度の水準は両版で同じでした。一方、FreSca の設定が意図どおり結果に反映されるようになりました。
- Forge Neo では、直前の SkimmedCFG または DifferenceCFG のフックが作った結果を、APG が上書きしなくなりました。SkimmedCFG と DifferenceCFG どうしが合成されない既存の制限は変わっていません。
- XYZ Grid で APG を無効にしたマスの infotext に、APG の値が書き込まれなくなりました。
- Momentum の説明を、新しい実測に基づいて書き直しました。以前の版では「多段サンプラーでは Momentum の値と結果が対応しない」と説明していましたが、その根拠の測定は削除済みの Adaptive Momentum（v2.0 で削除）で取ったもので、Momentum そのものには当てはまりません。

---

## デバッグ出力

WebUI を起動する前に、環境変数を設定します。

```powershell
$env:SD_WEBUI_SETI_DEBUG = "1"
```

コンソールに、登録したフックと、生成ごとに 1 回、実際の post-CFG の実行順が表示されます。

```text
[APG] registered post-CFG hook (reForge / Forge Classic backend), priority=15.4
[APG] post-CFG chain: fresca_hook(15.2) -> _apg_post_cfg_fn(15.4) -> _mahiro_fn(15.5)
```

---

## 動作原理

APG は、前に動いた拡張機能が作った CFG の押しを、条件付き（ポジティブ）予測を基準に取り出し、論文の Algorithm 1 を適用します。

```text
G         = denoised - cond                (ここまでに作られた押し)
diff      = G / (cond_scale - 1)           (論文の cond - uncond の大きさに戻す)
diff      = momentum_buffer.update(diff)   (Momentum != 0 のときのみ)
diff      = || diff ||_2 を norm_threshold に制限   (Norm Threshold > 0 のときのみ)
par, orth = diff を cond に平行な成分と直交する成分に分解
update    = orth + eta * par
out       = cond + (cond_scale - 1) * update
```

平行な成分は、画像がすでに向いている方向と同じ向きです。CFG でこれを増幅すると色が焼けるのはこのためです。APG より前に他の post-CFG 拡張機能がないときは、`diff` はちょうど `cond - uncond` になり、結果は論文の式そのものです。

### ComfyUI 組み込みノードとの違い

| | 本拡張機能 | ComfyUI `APG` ノード |
| --- | --- | --- |
| 最終合成 | `cond + (cond_scale - 1) * update`（論文） | 実質 `cond + cond_scale * update` |
| 縮約する次元 | バッチ以外の全次元 | 固定の `dim=[-1, -2, -3]` |
| 射影の精度 | 倍精度で計算して戻す（論文） | 入力の dtype のまま |
| Momentum のリセット | サンプリングごとに新しいバッファ | sigma が増えるたびに破棄 |

1 つ目の違いにより、中立の設定が通常の CFG と一致します。ComfyUI ノードはどの設定でもそうなりません。2 つ目は、5 次元の Anima / NextDiT の潜在表現でも論文どおり画像ごとに計算するためのもので、HuggingFace diffusers も同じ選択をしています。sigma の増加でリセットしないのは、可変ステップのサンプラーが棄却したステップを大きい sigma でやり直す正当な動作で、誤ってリセットされるためです。

---

# License / Acknowledgements / References

## License / ライセンス

sd-webui-APG is released under the MIT License. See [`LICENSE`](LICENSE).

本拡張機能は MIT License で公開しています。全文は [`LICENSE`](LICENSE) を参照してください。

Copyright (c) 2026 seti9585

---

## Acknowledgements / 謝辞

**Shiba-2-shiba**

The author first learned of APG through the note.com articles of [Shiba-2-shiba](https://note.com/gentle_murre488), whose [TCFG-APG-Mahiro-for-ForgeClassic](https://github.com/Shiba-2-shiba/TCFG-APG-Mahiro-for-ForgeClassic) implementation for Forge Classic was also consulted. Development of these extensions started from that work.

APG の存在は [Shiba-2-shiba](https://note.com/gentle_murre488) 氏の note.com の記事で知りました。Forge Classic 向けの実装 [TCFG-APG-Mahiro-for-ForgeClassic](https://github.com/Shiba-2-shiba/TCFG-APG-Mahiro-for-ForgeClassic) も参考にさせていただきました。一連の拡張機能の開発は、この記事と実装をきっかけに始まりました。深く感謝します。

**ComfyUI**

The built-in `APG` node of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (`comfy_extras/nodes_apg.py`, GPL-3.0) was read as a working reference for how APG is wired into a CFG hook. No code was carried over; the algorithm follows the paper, and the differences are listed above.

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) の組み込み `APG` ノード（`comfy_extras/nodes_apg.py`、GPL-3.0）は、APG を CFG のフックに組み込む方法の参考として読みました。コードの流用はありません。アルゴリズムは論文に従っており、違いは上に記載したとおりです。

---

## References / 典拠

Sadat, S., Hilliges, O., & Weber, R. M.  
*Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models.*  
ICLR 2025. [arXiv:2410.02416](https://arxiv.org/abs/2410.02416)

The algorithm follows Algorithm 1 of the paper: the `MomentumBuffer`, the double-precision projection, the scalar-zero initial `running_average` and the `(guidance_scale - 1)` final combination.

アルゴリズムは論文の Algorithm 1 に従っています（`MomentumBuffer`、倍精度での射影、履歴 `running_average` のスカラー 0 初期化、`(guidance_scale - 1)` による最終合成）。

Reference implementations / 参考実装:

- ComfyUI built-in `APG` node (`comfy_extras/nodes_apg.py`)
- HuggingFace diffusers `AdaptiveProjectedGuidance`
