# OnnxInterp
Onnx runtime interpreter for Elixir.
It is a Deep Learning inference framework that can be used in the same way as my TflInterp.

## Platform
I have confirmed it works in the following OS environment.

- Windows 10 with Visual C++ 2019
- WSL2/Ubuntu 20.04
- Linux Mint 20 "Ulyana"

## Requirements
cmake 3.13 or later is required.

Visual C++ 2019 for Windows.

## Installation
OnnxInterp is still a trial version, so please install it from my GitHub.
You can install it by adding `onnx_interp` to the `mix.exs` dependency list as follows:

```elixir
def deps do
  [
    {:onnx_interp, github: "shoz-f/onnx_interp"}
  ]
end
```

## Basic Usage
You get the trained Onnx model and save it in a directory that your application can read.
"your-app/priv" may be good choice.

```
$ cp your-trained-model.onnx ./priv
```

Next, you will create a module that interfaces with the deep learning model. The module will need pre-processing and
post-processing in addition to inference processing, as in the example following. OnnxInterp provides inference processing only.

You put `use OnnxInterp` at the beginning of your module, specify the model path as an optional argument. In the inference
section, you will put data input to the model (`OnnxInterp.set_input_tensor/3`), inference execution (`OnnxInterp.invoke/1`),
and inference result retrieval (`OnnxInterp.get_output_tensor/2`).

```elixr:your_model.ex
defmodule YourApp.YourModel do
  use OnnxInterp, model: "priv/your-trained-model.onnx"

  def predict(data) do
    # preprocess
    #  to convert the data to be inferred to the input format of the model.
    input_bin = convert-float32-binaries(data)

    # inference
    #  typical I/O data for Onnx models is a serialized 32-bit float tensor.
    output_bin =
      __MODULE__
      |> OnnxInterp.set_input_tensor(0, input_bin)
      |> OnnxInterp.invoke()
      |> OnnxInterp.get_output_tensor(0)

    # postprocess
    #  add your post-processing here.
    #  you may need to reshape output_bin to tensor at first.
    tensor = output_bin
      |> Nx.from_binary({:f, 32})
      |> Nx.reshape({size-x, size-y, :auto})

    * your-postprocessing *
    ...
  end
end
```

## Demo
There is Fast Neural Style Transfer: Candy demo in the demo directory.
This demo artistically converts a photo of a frog 'flog.jpg' in the demo directory and saves it as 'candy_flog.jpg'.

First, you download the trained DNN model "candy-9.onnx" from the following URL and place it in the demo directory.

- [candy-9.onnx: https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/candy-9.onnx](https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/candy-9.onnx)

You can run the demo by following these steps.

```shell
$ cd demo
$ mix deps.get
$ mix run -e "CandyDemo.demo"
```

Let's enjoy ;-)

## License
TflInterp is licensed under the Apache License Version 2.0.
