defmodule RetinaFace do
  import Nx.Defn

  @width  640
  @height 640

  alias OnnxInterp, as: NNInterp
  use NNInterp,
    model: "./model/retinaface_resnet50.onnx",
    url: "https://github.com/shoz-f/onnx_interp/releases/download/models/retinaface_resnet50.onnx",
    inputs: [f32: {1,3,@height,@width}],
    outputs: [f32: {1,16800,4}, f32: {1,16800,2}, f32: {1,16800,10}]

  def apply(img) do
    # preprocess
    bin = CImg.builder(img)
      |> CImg.resize({@width, @height}, :ul, 0)
      |> CImg.to_binary([{:gauss, {{104.0, 1.0}, {117.0, 1.0}, {123.0, 1.0}}}, :nchw])

    # prediction
    outputs = session()
      |> NNInterp.set_input_tensor(0, bin)
      |> NNInterp.invoke()

    [loc, conf, landm] = Enum.with_index([4, 2, 10], fn dim,i ->
        NNInterp.get_output_tensor(outputs, i) |> Nx.from_binary(:f32) |> Nx.reshape({:auto, dim})
      end)

    # postprocess
    scores = decode_scores(conf)
    boxes  = decode_boxes(loc)

    {:ok, res} = NNInterp.non_max_suppression_multi_class(__MODULE__,
        Nx.shape(scores), Nx.to_binary(boxes), Nx.to_binary(scores),
        iou_threshold: 0.4, score_threshold: 0.2,
        boxrepr: :corner)

    {:ok, fit2image_with_landmark(landm, res["0"], inv_aspect(img))}
  end


  @priorbox PostDNN.priorbox({@width, @height}, [{8, [16,32]}, {16, [64,128]}, {32, [256,512]}], [:transpose, :normalize])
  @variance Nx.tensor([0.1, 0.1, 0.2, 0.2], type: :f32) |> Nx.reshape({4,1})

  defnp decode_scores(conf) do
    Nx.slice_along_axis(conf, 1, 1, axis: 1)
  end

  defp decode_boxes(loc) do
    loc = Nx.transpose(loc)

    # decode box center coordinate on {1.0, 1.0}
    center = loc[0..1]
      |> Nx.multiply(@variance[0..1])
      |> Nx.multiply(@priorbox[2..3]) # * prior_size(x,y)
      |> Nx.add(@priorbox[0..1])      # + grid(x,y)

    # decode box half size
    half_size = loc[2..3]
      |> Nx.multiply(@variance[2..3])
      |> Nx.exp()
      |> Nx.multiply(@priorbox[2..3]) # * prior_size(x,y)
      |> Nx.divide(2.0)

    # decode boxes
    [Nx.subtract(center, half_size), Nx.add(center, half_size)]
      |> Nx.concatenate()
      |> PostDNN.clamp({0.0, 1.0})
      |> Nx.transpose()
  end

  defp fit2image_with_landmark(landm, nms_res, {inv_x, inv_y} \\ {1.0, 1.0}) do
    Enum.map(nms_res, fn [score, x1, y1, x2, y2, index] ->
      priorbox = Nx.slice_along_axis(@priorbox, index, 1, axis: 1) |> Nx.squeeze()
      variance = Nx.squeeze(@variance[0..1])

      landmark = landm[index]
        |> Nx.reshape({:auto, 2})
        |> Nx.multiply(variance)
        |> Nx.multiply(priorbox[2..3]) # * prior_size(x,y)
        |> Nx.add(priorbox[0..1])      # + grid(x,y)
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
