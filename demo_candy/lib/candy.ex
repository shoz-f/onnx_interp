# select dnn model in {"mosaic", "candy", "rain", "udnie", "point"} at compile time.
{model_name, model_path, model_url} =
  System.get_env("MODEL", "candy") |> String.downcase()
  |> tap(&IO.puts("-- COMPLE the deme for \"#{&1}\"."))
  |> case do
    "mosaic" -> {"Mosaic", "./model/mosaic-9.onnx", "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/mosaic-9.onnx"}
    "candy"  -> {"Candy", "./model/candy-9.onnx", "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/candy-9.onnx"}
    "rain"   -> {"RainPrincess", "./model/rain-princess-9.onnx", "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/rain-princess-9.onnx"}
    "udnie"  -> {"Udnie", "./model/udnie-9.onnx", "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/udnie-9.onnx"}
    "point"  -> {"Pointilism", "./model/pointilism-9.onnx", "https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/pointilism-9.onnx"}
    unknown -> raise("** COMILE ERROR: unknon dnn model: '#{unknown}'. it must be one of {vit, vgg16, resnet18} **")
  end


defmodule Candy do
  @moduledoc """
  Original work:
    Fast Neural Style Transfer - https://github.com/onnx/models/tree/main/vision/style_transfer/fast_neural_style
    fast-neural-style in PyTorch - https://github.com/pytorch/examples/tree/main/fast_neural_style
  """

  @name   model_name
  @model  model_path
  @url    model_url

  alias OnnxInterp, as: NNInterp
  use NNInterp, model: @model, url: @url

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

  def info() do
    %{name: @name, path: @model, url: @url}
  end
end
