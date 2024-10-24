defmodule YOLOP do
  @width  640
  @height 640

  alias OnnxInterp, as: NNInterp
  use OnnxInterp,
    model: "model/yolop-640-640.onnx",
    url: "https://github.com/hustvl/YOLOP/raw/main/weights/yolop-640-640.onnx",
    inputs: [f32: {1,3,@height,@width}],
    outputs: [f32: {1,25200,6}, f32: {1,2,@height,@width}, f32: {1,2,@height,@width}]

  def apply(img) do
    # preprocess
    input0 = img
      |> CImg.resize({@width, @height})
      |> CImg.to_binary([{:gauss, {{103.53,57.375},{116.28,57.12},{123.675,58.395}}}, :nchw])

    # prediction
    outputs = session()
      |> OnnxInterp.set_input_tensor(0, input0)
      |> OnnxInterp.invoke()

    det_out        = OnnxInterp.get_output_tensor(outputs, 0) |> Nx.from_binary(:f32) |> Nx.reshape({:auto, 6})
    drive_area_seg = OnnxInterp.get_output_tensor(outputs, 1) |> Nx.from_binary(:f32) |> Nx.reshape({2, :auto})
    lane_line_seg  = OnnxInterp.get_output_tensor(outputs, 2) |> Nx.from_binary(:f32) |> Nx.reshape({2, :auto})

    # postprocess
    img_size = CImg.shape(img) |> then(fn {w,h,_,_} -> {w,h} end)
    {
      :ok,
      decode_bbox(det_out, img_size),
      decode_segments(drive_area_seg, img_size),
      decode_segments(lane_line_seg, img_size)
    }
  end

  def decode_bbox(t, {w,h}) do
    boxes  = Nx.slice_along_axis(t, 0, 4, axis: 1)
    scores = Nx.multiply(Nx.slice_along_axis(t, 4, 1, axis: 1), (Nx.slice_along_axis(t, 5, 1, axis: 1)))

    {:ok, res} = NNInterp.non_max_suppression_multi_class(__MODULE__,
      Nx.shape(scores), Nx.to_binary(boxes), Nx.to_binary(scores),
      iou_threshold: 0.45, score_threshold: 0.25
    )

    scale_x = fn x -> round(x * w / @width) end
    scale_y = fn y -> round(y * h / @height) end
    Enum.map(res["0"], fn [score, x1, y1, x2, y2, index] ->
      [score, scale_x.(x1), scale_y.(y1), scale_x.(x2), scale_y.(y2), index]
    end)
  end

  defp decode_segments(t, img_size) do
    Nx.greater(t[1], t[0])
    |> Nx.to_binary()
    |> CImg.from_binary(640, 640, 1, 1, dtype: "<u1")
    |> CImg.resize(img_size)
  end
end
