defmodule Demo do
  use OnnxInterp, model: "./.onnx", label: "./coco.label"

  @_shape {640, 480}

  def apply(img) do
    # preprocess
    bin = img
      |> CImg.resize(@_shape)
      |> CImg.to_binary()

    # prediction
    outputs =
      __MODULE__
      |> OnnxInterp.set_input_tensor(0, bin)
      |> OnnxInterp.invoke()
      |> OnnxInterp.get_output_tensor(0)
      |> Nx.from_binary({:f, 32}) |> Nx.reshape({:auto, })

    # postprocess
  end

  defp scale(img) do
    {w, h, _, _}   = CImg.shape(img)
    {wsize, hsize} = @_shape
    max(w/wsize, h/hsize)
  end
end
