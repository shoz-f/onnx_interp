defmodule Candy do
  @moduledoc """
  Original work:
    Fast Neural Style Transfer - https://github.com/onnx/models/tree/main/vision/style_transfer/fast_neural_style
    fast-neural-style in PyTorch - https://github.com/pytorch/examples/tree/main/fast_neural_style
  """

  alias OnnxInterp, as: NNInterp
  use NNInterp,
    model: "./model/candy-9.onnx",
    url: "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/candy-9.onnx"

  def apply(img) do
    input = img
      |> CImg.resize({224, 224})
      |> CImg.to_binary([{:dtype, "<f4"}, {:range, {0.0, 255.0}}, :nchw])

    __MODULE__
    |> NNInterp.set_input_tensor(0, input)
    |> NNInterp.invoke()
    |> NNInterp.get_output_tensor(0)
    |> CImg.from_binary(224, 224, 1, 3, [{:dtype, "<f4"}, {:range, {0.0, 255.0}}, :nchw])
  end
end
