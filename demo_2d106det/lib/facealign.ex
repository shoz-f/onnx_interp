defmodule FaceAlign do
  @width  192
  @height 192

  alias OnnxInterp, as: NNInterp
  use NNInterp,
    model: "model/2d106det.onnx",
    url: "https://github.com/shoz-f/onnx_interp/releases/download/models/2d106det.onnx",
    inputs: [f32: {1,3,@height,@width}],
    outputs: [f32: {1,212}]

  def apply(img) do
    # preprocess
    input0 = img
      |> CImg.resize({@width, @height})
      |> CImg.to_binary([{:range, {0.0, 255.0}}, :nchw])

    # prediction
    output0 = session()
      |> NNInterp.set_input_tensor(0, input0)
      |> NNInterp.invoke()
      |> NNInterp.get_output_tensor(0)
      |> Nx.from_binary(:f32) |> Nx.reshape({:auto, 2})

    # postprocess
    landmark = output0
      |> Nx.add(Nx.tensor([1.0, 1.0]))
      |> Nx.divide(2.0)
      |> Nx.to_flat_list()
      |> Enum.chunk_every(2)

    {:ok, landmark}
  end
end
