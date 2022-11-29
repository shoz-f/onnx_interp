defmodule NanoDet do
  @moduledoc """
  Original work
    NanoDet-Plus - https://github.com/RangiLyu/nanodet
  """

  alias OnnxInterp, as: NNInterp
  use NNInterp, label: "./model/coco.label",
    model: "./model/NanoDet-Plus-m-416.onnx",
    url: "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416.onnx"

  @width  416
  @height 416
  @grid   PostDNN.meshgrid({@width, @height}, [8, 16, 32, 64], [:normalize])
  @arm    Nx.iota({8})

  def apply(img) do
    # preprocess
    input0 = img
      |> CImg.resize({@width, @height})
      |> CImg.to_binary([{:gauss, {{103.53, 57.375}, {116.28, 57.12}, {123.675, 58.395}}}, :nchw, :bgr])

    # prediction
    output0 = __MODULE__
      |> OnnxInterp.set_input_tensor(0, input0)
      |> OnnxInterp.invoke()
      |> OnnxInterp.get_output_tensor(0)
      |> Nx.from_binary(:f32) |> Nx.reshape({:auto, 112})

    # postprocess
    scores = Nx.slice_along_axis(output0,  0, 80, axis: 1)
    boxes  = Nx.slice_along_axis(output0, 80, 32, axis: 1)

    # seive candidates
    [scores, boxes, grid] = PostDNN.sieve(scores, [boxes, @grid], fn t1 ->
      Nx.reduce_max(t1, axes: [1]) |> Nx.greater_equal(0.25)
    end)

    boxes = decode_boxes(boxes, grid)

    OnnxInterp.non_max_suppression_multi_class(__MODULE__,
      Nx.shape(scores), Nx.to_binary(boxes), Nx.to_binary(scores), boxrepr: :corner
    )
  end

  defp decode_boxes(boxes, grid) do
    [grid_x, grid_y, pitch] =
      Enum.map(0..2, fn i ->
        Nx.slice_along_axis(grid, i, 1, axis: 1) |> Nx.squeeze()
      end)

    Enum.map(0..3, fn i ->
      exp  = Nx.slice_along_axis(boxes, 8 * i, 8, axis: 1) |> Nx.exp()
      wing = Nx.dot(exp, @arm) |> Nx.divide(Nx.sum(exp, axes: [1])) |> Nx.multiply(pitch)

      case i do
        0 -> Nx.subtract(grid_x, wing) |> Nx.max(0.0)
        1 -> Nx.subtract(grid_y, wing) |> Nx.max(0.0)
        2 -> Nx.add(grid_x, wing) |> Nx.min(1.0)
        3 -> Nx.add(grid_y, wing) |> Nx.min(1.0)
      end
      |> Nx.reshape({:auto, 1})
    end)
    |> Nx.concatenate(axis: 1)
  end
end
