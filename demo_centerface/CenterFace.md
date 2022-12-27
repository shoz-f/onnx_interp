# 0.Prologue
暇つぶしに、興味を引いた DNNアプリを *Interpに移植して遊んでいる。
本稿はその雑記&記録。

先日、顔検出の「RetinaFace」と言うモデルを OnnxInterpに移植して遊んでみた訳だが、世間の下馬評のごとく顔の検出力は良いもののレスポンスが少し遅いなぁと感じる。そんな弱点を埋めようと、「YuNet」や「Ultra-Light-Fast-Generic-Face-Detector」などが雨後の筍のように現れたようだ。今回移植してみる「CenterFace」もその１つ。

# 1.Original Work
「CenterFace」の特徴は、

手始めに、上の記事が紹介している"YuNet"を移植してみたが、検出力がいまいちであった。それでは面白くないので、あれこれと「顔検出」モデルを調べてみたところ、「RetinaFace」と言うモデルの評判が良さそうだと分かった。これを移植することにする。

RetinaFace: Single-stage Dense Face Localisation in the Wild
- https://arxiv.org/abs/1905.00641

RetinaFace in PyTorch
- https://github.com/biubug6/Pytorch_Retinaface

本家本元はMxnetベースらしいので *Interpへの移植は難しい。PyTorchに移植したプロジェクト「RetinaFace in PyTorch」があったので、ここから ONNXモデルを作成することにする。

RetinaFaceのモデル・アーキテクチャは下図の様になっている。水色の部分は右隣の Feature Pyramidの入力となる特徴抽出ブロックのようで、ResNetや MobileNetを参照しているようだ。Feature Pyramidとは独立した Context Moduleで、Face class/Face box/Facial landmark/Dense Regressionの確率を計算しているらしい。最後の Dense Regressionは、グリッドまたはアンカーボックスに関する確率ってことかな。

![スクリーンショット 2022-12-24 173657.jpg](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/14158/88bca221-fd24-3fea-c9a5-7ba39b2e3904.jpeg)

(抜粋:「RetinaFace: Single-stage Dense Face Localisation in the Wild」より)

# 2.準備
RetinaFaceの ONNXモデルは、上の「RetinaFace in PyTorch」プロジェクトから調達する。

プロジェクトを git cloneし、下記の googleドライブから学習済みweightをダウンロードする。学習済みweightの格納先は weightsディレクトリとする。

> https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1

```shell
Pytorch_Retinaface/
└── weights
     ├── Resnet50_Final.pth
     ├── mobilenet0.25_Final.pth
     └── mobilenetV1X0.25_pretrain.tar
```

PyTorchからONNXへのコンバートは、プロジェクトに添付の `convert_to_onnx.py`スクリプトを利用する。引数無しでスクリプトを実行すると、バックボーンには MobileNetが配置される。バックボーンに ResNetを配置するには、次の様にする。

```python
python convert_to_onnx.py -m ./weights/Resnet50_Final.pth --network resnet50 --cpu
```

作成される ONNXモデルのファイル名は 'FaceDetector.onnx'に固定されている。以下の OnnxInterpアプリでは、これを'retinaface_resnet50.onnx'または'retinaface_mobile0.25.onnx'にリネームして使用する。

# 3.OnnxInterp用のLivebookノート

Mix.installの依存リストに記述するモジュールは下記の通り。RetinaFaceではアンカーボックスが必要なので、PostDNNを含めている。

```elixir: setup cell
File.cd!(__DIR__)
# for windows JP
System.shell("chcp 65001")

Mix.install([
  {:onnx_interp, path: ".."},
  {:cimg, "~> 0.1.16"},
  {:postdnn, "~> 0.1.4"},
  {:nx, "~> 0.4.0"},
  {:kino, "~> 0.7.0"}
])
```

入力画像のresizeは、aspect比保存で行う(:ulオプション)。floatへの型変換は初めてみるタイプだが、平均値{R:104.0,G:117.0,B:123.0},分散1.0と解釈すれば、ガウス分布タイプの正規化で代用できる。

後処理では、顔のスコアとデコードしたBBoxを NMSに掛けて、検出した顔のBBoxを得る。背景のスコアは使用しない。また、モデルが出力するBBoxはアンカーボックスに対する相対座標・比寸法で表されているので、元のグリッドの座標とアンカーボックスの寸法を掛け合わせて画像座標系におけるBBoxに変換する(decode_boxes/1)。この処理は、YuNetのそれと全く同じだった。

NMS(OnnxInterp.non_max_suppression_multi_class/4)の出力は fit2image_with_landmark/4に通し、各座標値を入力画像の座標系に戻す逆aspect変換(?)を行う。また、同時にランドマークのデコード済み座標を添付する。

ランドマークのデコードは、NMSで残ったBBoxに紐づくモノだけを対象とする。もしもNMS前にデコードを行うと、全てのランドマークをデコードすることになり無駄な計算処理が発生する。*Interp, PostDNNに実装しているNMSは、スコアとBBoxそして元のテーブルにおけるインデックスを返すので、インデックスを手掛かりに紐づくランドマークのレコードを見つけることができる。

apply/1の出力は、次のリストを要素とするリスト。

```elixir
[スコア,BBox左上X,BBox左上Y,BBox右下X,BBox右下Y,[[ランドマーク1X,ランドマーク1Y],..,[ランドマーク5X,ランドマーク5Y]]]
```


> [モデル・カード]
> - inputs:
> [0] f32:{1,3,640,640} - RGB画像,NCHWレイアウト,各画素は右式で変換 R'=float(R-104),G'=float(G-117), B'=float(B-123)
> - outputs:
> [0] f32:{1,16800,4} - BBox(中心X,中心Y,サイズX,サイズY), アンカーボックスに対する相対座標、比寸法
> [1] f32:{1,16800,2} - スコア(背景,顔)
> [2] f32:{1,16800,10} - ランドマーク(Xi,Yi) x 5, アンカーボックスに対する比率で表現
>
> - prioribox:
> 格子間隔 8,16,32のグリッドそれぞれに 2個のアンカーボックス, サイズ@8[16,32],@16[64,128],@32[256,512]

```elixir: centerface
defmodule CenterFace do
  import Nx.Defn

  @width  640
  @height 640

  alias OnnxInterp, as: NNInterp
  use NNInterp,
    model: "./model/centerface_dynamic.onnx",
    url: "https://github.com/shoz-f/onnx_interp/releases/download/models/centerface_dynamic.onnx",
    inputs: [f32: {1,3,@height,@width}],
    outputs: [f32: {1,1,div(@height,4),div(@width,4)}, f32: {1,2,div(@height,4),div(@width,4)}, f32: {1,2,div(@height,4),div(@width,4)}, f32: {1,10,div(@height,4),div(@width,4)}]

  def apply(img) do
    # preprocess
    bin = CImg.builder(img)
      |> CImg.resize({@width, @height}, :ul, 0)
      |> CImg.to_binary([{:range, {0.0, 255.0}}, :nchw])

    # prediction
    outputs = session()
      |> NNInterp.set_input_tensor(0, bin)
      |> NNInterp.invoke()

    [heatmap, scale, offset, landm] = Enum.with_index([1, 2, 2, 10], fn dim,i ->
        NNInterp.get_output_tensor(outputs, i) |> Nx.from_binary(:f32) |> Nx.reshape({dim, :auto})
      end)

    # postprocess
    scores = Nx.transpose(heatmap)
    boxes  = decode_boxes(offset, scale)
    landm  = Nx.transpose(landm)

    {:ok, res} = NNInterp.non_max_suppression_multi_class(__MODULE__,
        Nx.shape(scores), Nx.to_binary(boxes), Nx.to_binary(scores),
        iou_threshold: 0.2, score_threshold: 0.2,
        boxrepr: :corner)

    {:ok, fit2image_with_landmark(landm, res["0"], inv_aspect(img))}
  end


  @grid PostDNN.meshgrid({@width, @height}, [4], [:center, :normalize, :transpose])

  defp decode_boxes(offset, size) do
    # decode box center coordinate on {1.0, 1.0}
    center = offset
      |> Nx.reverse(axes: [0])     # swap (y,x) -> (x,y)
      |> Nx.multiply(@grid[2..3]) # * grid_pitch(x,y)
      |> Nx.add(@grid[0..1])      # + grid(x,y)

    # decode box half size
    half_size = size
      |> Nx.reverse(axes: [0])     # swap (y,x) -> (x,y)
      |> Nx.exp()
      |> Nx.multiply(@grid[2..3]) # * grid_pitch(x,y)
      |> Nx.divide(2.0)

    # decode boxes
    [Nx.subtract(center, half_size), Nx.add(center, half_size)]
      |> Nx.concatenate()
      |> PostDNN.clamp({0.0, 1.0})
      |> Nx.transpose()
  end

  defp fit2image_with_landmark(landm, nms_res, {inv_x, inv_y} \\ {1.0, 1.0}) do
    Enum.map(nms_res, fn [score, x1, y1, x2, y2, index] ->
      grid = Nx.slice_along_axis(@grid, index, 1, axis: 1) |> Nx.squeeze()

      landmark = landm[index]
        |> Nx.reshape({:auto, 2})
        |> Nx.reverse(axes: [0])
        |> Nx.multiply(grid[2..3]) # * prior_size(x,y)
        |> Nx.add(grid[0..1])      # + grid(x,y)
        |> Nx.multiply(Nx.tensor([inv_x, inv_y]))
        |> Nx.to_flat_list()
        |> Enum.chunk_every(2)

      [score, x1*inv_x, y1*inv_y, x2*inv_x, y2*inv_y, landmark]
    end)
  end

  defp inv_aspect(img) do
    {w, h, _, _} = CImg.shape(img)
    if w > h, do: {1.0, w / h}, else: {h / w, 1.0}
  end
end
```

デモ・モジュール DemoRetinaFaceには、引数に与えた一枚の画像に推論を掛けその結果を表示する run/1を用意する。

推論結果の描画は、下請け関数 draw_item/3が行う。ランドマークの描画は、今日時点では行わない[*1]。

[*1]CImgに draw_makerのような機能追加するまでお預け。

```elixir: demo_centerface
defmodule DemoCenterFace do
  def run(path) do
    img = CImg.load(path)

    with {:ok, res} = CenterFace.apply(img) do
      res
      |> draw_item(CImg.builder(img), {0, 255, 0})
      |> CImg.display_kino(:jpeg)
    end
  end

  defp draw_item(boxes, canvas, color \\ {255, 255, 255}) do
    Enum.reduce(boxes, canvas, fn [_score, x1, y1, x2, y2, _landmark], canvas ->
      CImg.fill_rect(canvas, x1, y1, x2, y2, color, 0.3)
    end)
  end
end
```

# 4.デモンストレーション

`CenterFace`を起動する。

```elixir
CenterFace.start_link([])
```

画像を与え、顔検出を行う。

```elixir
DemoCenterFace.run("10.jpg")
```

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/14158/02990743-34ef-67a2-1d78-abb2602dcbbc.png)

# 5.Epilogue
PythonにはDNNベースの画像認識/検出を集めてライブラリ化しているプロジェクトがある。

- InsightFace: 2D and 3D Face Analysis Project
https://github.com/deepinsight/insightface

- deepface
https://github.com/serengil/deepface

Elixirでそー言うものに取り組んでも面白そうだなぁと思う今日この頃。<br>
とりあえず、もう二つ三つ画像検出or画像認識を移植して遊んでみようか。

(END)

# Appendix
- OnnxInterpのノート
https://github.com/shoz-f/onnx_interp/blob/main/demo_retinaface/RetinaFace.livemd
