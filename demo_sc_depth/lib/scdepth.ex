defmodule ScDepth do
  @width  640
  @height 384

  alias OnnxInterp, as: NNInterp
  use NNInterp,
    model: "model/sc_depth.onnx",
    inputs: [f32: {1,3,@height,@width}],
    outputs: [f32: {1,3,@height,@width}]

  def apply(img) do
    # preprocess
    input0 = img
      |> CImg.resize({@width, @height})
      |> CImg.to_binary([{:gauss, {{114.75,57.375},{114.75,57.375},{114.75,57.375}}}, :nchw])

    # prediction
    output0 = session()
      |> NNInterp.set_input_tensor(0, input0)
      |> NNInterp.invoke()
      |> NNInterp.get_output_tensor(0)

    # postprocess
    output0
    |> CImg.from_binary(@width, @height, 1, 1, range: min_max(output0), dtype: "<f4")
    |> CImg.color_mapping(:hot)
  end
  
  defp min_max(bin) do
    t = Nx.from_binary(bin, :f32)
    {
      Nx.reduce_min(t) |> Nx.to_number(),
      Nx.reduce_max(t) |> Nx.to_number()
    }
  end
end
