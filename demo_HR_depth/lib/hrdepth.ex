defmodule HRDepth do
  def apply(img) do
    img
    |> HRDepthEncoder.apply()
    |> HRDepthDecoder.apply()
  end
end

defmodule HRDepthEncoder do
  @width  1280
  @height 384

  alias OnnxInterp, as: NNInterp
  use NNInterp,
    model: "model/HR_Depth_K_M_1280x384_encoder.onnx",
    url: "https://github.com/shoz-f/onnx_interp/releases/download/models/HR_Depth_K_M_1280x384_encoder.onnx",
    inputs: [f32: {1,3,@height,@width}],
    outputs: [f32: {1,64,192,640}, f32: {1,64,96,320}, f32: {1,128,48,160}, f32: {1,256,24,80}, f32: {1,512,12,40}]

  def apply(img) do
    # preprocess
    input0 = img
      |> CImg.resize({@width, @height})
      |> CImg.to_binary([{:range, {0.0, 1.0}}, :nchw])

    # prediction
    session()
    |> NNInterp.set_input_tensor(0, input0)
    |> NNInterp.invoke()
    |> get_output_tensors(0..4)
  end

  def get_output_tensors(session, range) do
    for i <- range, do: NNInterp.get_output_tensor(session, i)
  end
end

defmodule HRDepthDecoder do
  @width  1280
  @height 384

  alias OnnxInterp, as: NNInterp
  use NNInterp,
    model: "model/HR_Depth_K_M_1280x384_decoder.onnx",
    url: "https://github.com/shoz-f/onnx_interp/releases/download/models/HR_Depth_K_M_1280x384_decoder.onnx",
    inputs: [f32: {1,64,192,640}, f32: {1,64,96,320}, f32: {1,128,48,160}, f32: {1,256,24,80}, f32: {1,512,12,40}],
    outputs: [f32: {1,1,@height,@width}, f32: {1,1,192,640}, f32: {1,1,96,320}, f32: {1,1,48,160}]

  def apply(inputs) do
    # prediction
    output0 = session()
      |> set_input_tensors(inputs)
      |> NNInterp.invoke()
      |> NNInterp.get_output_tensor(0)

    # postprocess
    CImg.from_binary(output0, @width, @height, 1, 1, range: min_max(output0), dtype: "<f4")
  end

  def set_input_tensors(session, items, offset \\ 0) when is_list(items) do
    Enum.with_index(items, offset)
    |> Enum.reduce(session, fn {item, i}, session -> NNInterp.set_input_tensor(session, i, item) end)
  end

  defp min_max(bin) do
    t = Nx.from_binary(bin, :f32)
    {
      Nx.reduce_min(t) |> Nx.to_number(),
      Nx.reduce_max(t) |> Nx.to_number()
    }
  end
end
