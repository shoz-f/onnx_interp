defmodule DemoViT do
  alias OnnxInterp, as: NNInterp

  @width  384
  @height 384

  use NNInterp,
    model: "./data/vit.onnx",
    url: "https://github.com/shoz-f/onnx_interp/releases/download/0.1.2/vit.onnx"

  @imagenet1000 (for item <- File.stream!("./imagenet1000.label") do
                   String.trim_trailing(item)
                 end)
                |> Enum.with_index(&{&2, &1})
                |> Enum.into(%{})

  def apply(img, top \\ 1) do
    # preprocess
    input0 = CImg.builder(img)
      |> CImg.resize({@width, @height})
      |> CImg.to_binary([{:gauss, {{123.7, 58.4}, {116.3, 57.1}, {103.5, 57.4}}}, :nchw])

    # prediction
    output0 = __MODULE__
      |> NNInterp.set_input_tensor(0, input0)
      |> NNInterp.invoke()
      |> NNInterp.get_output_tensor(0)
      |> Nx.from_binary({:f, 32})
      |> Nx.reshape({1000})

    # postprocess
    exp = Nx.exp(output0)

    Nx.divide(exp, Nx.sum(exp))     # softmax
    |> Nx.argsort(direction: :desc)
    |> Nx.slice([0], [top])
    |> Nx.to_flat_list()
    |> Enum.map(&@imagenet1000[&1])
  end
  
  def run() do
    unless File.exists?("lion.jpg"),
      do: NNInterp.URL.download("https://github.com/shoz-f/nn-interp/releases/download/0.0.1/lion.jpg")

    CImg.load("lion.jpg")
    |> __MODULE__.apply(3)
    |> IO.inspect()
    :ok
  end
end
