defmodule UltraFace do
  @width  320
  @height 240

  alias OnnxInterp, as: NNInterp
  use NNInterp,
    model: "./model/version-slim-320_simplified.onnx",
    inputs: [f32: {1,3,@height,@width}],
    outputs: [f32: {1,4420,2}, f32: {1,4420,4}]

  @width  320
  @height 240

  def apply(img) do
    # preprocess
    input0 = img
      |> CImg.resize({@width, @height})
      |> CImg.to_binary([{:range, {-1.0, 1.0}}, :nchw])

    # prediction
    outputs = __MODULE__
      |> NNInterp.set_input_tensor(0, input0)
      |> NNInterp.invoke()
      
      
    conf = NNInterp.get_output_tensor(outputs, 0) |> Nx.from_binary(:f32) |> Nx.reshape({:auto, 2})
    loc  = NNInterp.get_output_tensor(outputs, 1) |> Nx.from_binary(:f32) |> Nx.reshape({:auto, 4})

#    loc  = NNInterp.get_output_tensor(outputs, 0) |> Nx.from_binary(:f32) |> Nx.reshape({:auto, 14})
#    conf = NNInterp.get_output_tensor(outputs, 1) |> Nx.from_binary(:f32) |> Nx.reshape({:auto,  2})
#    iou  = NNInterp.get_output_tensor(outputs, 2) |> Nx.from_binary(:f32) |> Nx.reshape({:auto,  1})
#
#    # postprocess
    scores = decode_scores(conf)
    boxes  = decode_boxes(loc)

    NNInterp.non_max_suppression_multi_class(__MODULE__,
      Nx.shape(scores), Nx.to_binary(boxes), Nx.to_binary(scores),
      iou_threshold: 0.3, score_threshold: 0.7,
      boxrepr: :corner
    )
#    |> PostDNN.adjust2letterbox(CImg.Util.aspect(img))
  end


  @priorbox PostDNN.priorbox({@width, @height}, [{8, [10,16,24]}, {16, [32,48]}, {32, [64,96]}, {64, [128,192,256]}], [:transpose, :normalize])
  @variance Nx.tensor([0.1, 0.1, 0.2, 0.2], type: :f32) |> Nx.reshape({4,1})

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
    boxes = [Nx.subtract(center, half_size), Nx.add(center, half_size)]
      |> Nx.concatenate()
      |> PostDNN.clamp({0.0, 1.0})
      |> Nx.transpose()

    # decode landmarks
#    landmarks = Enum.map([4,6,8,10,12], fn i ->
#        loc[i..i+1]
#        |> Nx.multiply(@variance[0..1])
#        |> Nx.multiply(@priorbox[2..3])
#        |> Nx.add(@priorbox[0..1])
#      end)
#      |> Nx.concatenate()
#      |> Nx.transpose()

    boxes
  end
                                                                                                                                                                                                                         
  defp decode_scores(conf) do
    Nx.slice_along_axis(conf, 1, 1, axis: 1)
  end
end
