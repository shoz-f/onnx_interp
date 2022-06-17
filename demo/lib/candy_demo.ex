defmodule CandyDemo do
  @moduledoc """
  Documentation for `CandyDemo`.
  
  ### Prepare
  Download the pre-trained model of Candy style from here:
    https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/candy-9.onnx
  """

  use OnnxInterp, model: "candy-9.onnx"

  def apply(img) do
    input = CImg.builder(img)
      |> CImg.resize({224, 224})
      |> CImg.to_binary([{:dtype, "<f4"}, {:range, {0.0, 255.0}}, :nchw])

    session()
    |> OnnxInterp.set_input_tensor(0, input)
    |> OnnxInterp.run()
    |> OnnxInterp.get_output_tensor(0)
    |> CImg.from_binary(224, 224, 1, 3, [{:dtype, "<f4"}, {:range, {0.0, 255.0}}, :nchw])
  end

  def demo() do
    CImg.load("flog.jpg")
    |> apply()
    |> CImg.save("candy_flog.jpg")
  end
end
