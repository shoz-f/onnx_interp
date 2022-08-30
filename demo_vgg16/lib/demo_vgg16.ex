defmodule DemoVGG16 do
  use OnnxInterp, model: "./vgg16-7.onnx"

  @vgg16_shape {224, 224}
  @imagenet1000 (for item <- File.stream!("./imagenet1000.label") do
                    String.trim_trailing(item)
                  end)
                  |> Enum.with_index(&{&2, &1})
                  |> Enum.into(%{})

  def apply(img, top) do
    # preprocess
    bin = img
      |> CImg.resize(@vgg16_shape)
      |> CImg.to_binary([{:range, {-2.2, 2.7}}, :nchw])

    # prediction
    outputs = __MODULE__
      |> OnnxInterp.set_input_tensor(0, bin)
      |> OnnxInterp.invoke()
      |> OnnxInterp.get_output_tensor(0)
      |> Nx.from_binary({:f, 32}) |> Nx.reshape({1000})

    # postprocess
    outputs
    |> Nx.argsort(direction: :desc)
    |> Nx.slice([0], [top])
    |> Nx.to_flat_list()
    |> Enum.map(&@imagenet1000[&1])
  end
end
