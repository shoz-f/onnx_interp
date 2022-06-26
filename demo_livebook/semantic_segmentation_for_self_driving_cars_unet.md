# Semantic Segmentation for Self Driving Cars: UNet

```elixir
Mix.install([
  {:kino, "~> 0.6.1"},
  {:req, "~> 0.2.2"},
  {:nx, "~> 0.2.1"},
  {:cimg, github: "shoz-f/cimg_ex"},
  {:onnx_interp, path: "../onnx_interp"}
])

Req.get!("https://github.com/shoz-f/onnx_interp/releases/download/0.1.2/unet.onnx").body
|> then(fn x -> File.write!("unet.onnx", x) end)
```

## Inference module of UNet

```elixir
defmodule Unet do
  use OnnxInterp, model: "./unet.onnx"

  @unet_shape {512, 512}

  def apply(img) do
    # save original shape
    {w, h, _, _} = CImg.shape(img)

    # preprocess
    bin =
      CImg.builder(img)
      |> CImg.resize(@unet_shape)
      |> CImg.to_binary()

    # prediction
    outputs =
      session()
      |> OnnxInterp.set_input_tensor(0, bin)
      |> OnnxInterp.run()
      |> OnnxInterp.get_output_tensor(0)
      |> Nx.from_binary({:f, 32})
      |> Nx.reshape({512, 512, 32})
      |> Nx.argmax(axis: 2)
      |> Nx.as_type({:u, 8})

    # postprocess
    Nx.to_binary(outputs)
    |> CImg.from_binary(512, 512, 1, 1, [{:dtype, "<u1"}])
    |> CImg.resize({w, h})
  end
end

Unet.start_link([])
```

## Apply to a sample image

```elixir
input =
  Req.get!("https://github.com/shoz-f/onnx_interp/releases/download/0.1.2/sample.jpg").body
  |> CImg.from_binary()

mask =
  Unet.apply(input)
  |> CImg.color_mapping(:lines)

input
|> CImg.blend(mask)
|> CImg.resize(0.5)
|> CImg.display_kino(:jpeg)
```
