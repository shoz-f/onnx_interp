# select dnn model in {"vit", "vgg16", "resnet18"} at compile time.
{model_name, model_path, {model_width, model_height}, model_url} =
  System.get_env("MODEL", "resnet18") |> String.downcase()
  |> tap(&IO.puts("-- COMPLE the deme for \"#{&1}\"."))
  |> case do
    "vit"      -> {"ViT", "./model/vit.onnx", {384, 384}, "https://drive.google.com/uc??authuser=0&export=download&confirm=t&id=1L2sDNeK7CXdiMYflgCN8dWkrP2397ULv"}
    "vgg16"    -> {"VGG16", "./model/vgg16-7.onnx", {224, 224}, "https://github.com/onnx/models/raw/main/vision/classification/vgg/model/vgg16-7.onnx"}
    "resnet18" -> {"ResNet18", "./model/resnet18-v1-7.onnx", {224, 224}, "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet18-v1-7.onnx"}
    unknown -> raise("** COMILE ERROR: unknon dnn model: '#{unknown}'. it must be one of {vit, vgg16, resnet18} **")
  end


defmodule ImageClassify do
  @moduledoc """
  ## Original work:
    * ONNX Model Zoo/ResNet - https://github.com/onnx/models/tree/main/vision/classification/resnet
    * ONNX Model Zoo/VGG - https://github.com/onnx/models/tree/main/vision/classification/vgg
    * Vision Transformer (ViT) in PyTorch - https://github.com/lukemelas/PyTorch-Pretrained-ViT
  
  ## How to get pre-trained ViT onnx model.
    You need install "pytorch_pretrained_vit" and covert the model beforehand.

    '$ pip install pytorch_pretrained_vit'

    ```python:
    convert Pytorch ViT model to ONNX.

    import torch
    from pytorch_pretrained_vit import ViT

    model = ViT('B_16_imagenet1k', pretrained=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 384, 384)

    os.makedirs("data", exist_ok=True)
    torch.onnx.export(model,
      dummy_input,
      "vit.onnx",
      verbose=False,
      input_names=["input.0"],
      output_names=["output.0"],
      export_params=True
    )
    ```
  """

  @name   model_name
  @model  model_path
  @url    model_url
  @width  model_width
  @height model_height

  alias OnnxInterp, as: NNInterp
  use NNInterp, model: @model, url: @url

  @imagenet1000 (for item <- File.stream!("./model/imagenet1000.label") do
                    String.trim_trailing(item)
                  end)
                  |> Enum.with_index(&{&2, &1})
                  |> Enum.into(%{})

  def apply(img, top \\ 1) do
    # preprocess
    input0 = img
      |> CImg.resize({@width, @height})
      |> CImg.to_binary([{:gauss, {{123.7, 58.4}, {116.3, 57.1}, {103.5, 57.4}}}, :nchw])

    # prediction
    output0 = __MODULE__
      |> NNInterp.set_input_tensor(0, input0)
      |> NNInterp.invoke()
      |> NNInterp.get_output_tensor(0)
      |> Nx.from_binary(:f32) |> Nx.reshape({1000})

    # postprocess
    then(Nx.exp(output0), fn exp -> Nx.divide(exp, Nx.sum(exp)) end) # softmax
    |> Nx.argsort(direction: :desc)
    |> Nx.slice([0], [top])
    |> Nx.to_flat_list()
    |> Enum.map(&@imagenet1000[&1])
  end

  def info() do
    %{name: @name, path: @model, shape: {@width, @height}, url: @url}
  end
end
