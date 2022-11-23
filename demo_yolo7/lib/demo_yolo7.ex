defmodule DemoYolo7 do
  use OnnxInterp, model: "./yolov7.onnx", label: "./coco.label"

  @yolo7_shape {640, 480}

  def apply(img) do
    # preprocess
    bin = img
      |> CImg.resize(@yolo7_shape)
      |> CImg.to_binary([{:range, {0.0, 1.0}}, :nchw])

    # prediction
    outputs =
      __MODULE__
      |> OnnxInterp.set_input_tensor(0, bin)
      |> OnnxInterp.invoke()
      |> OnnxInterp.get_output_tensor(0)
      |> Nx.from_binary({:f, 32}) |> Nx.reshape({:auto, 85})

    # postprocess
    boxes  = extract_boxes(outputs, scale(img))
    scores = extract_scores(outputs)

    OnnxInterp.non_max_suppression_multi_class(__MODULE__,
      Nx.shape(scores), Nx.to_binary(boxes), Nx.to_binary(scores)
    )
    
    #{outputs, boxes, scores}
  end

  defp extract_boxes(tensor, scale) do
    Nx.slice_along_axis(tensor, 0, 4, axis: 1) |> Nx.multiply(scale)
  end

  defp extract_scores(tensor) do
    Nx.multiply(Nx.slice_along_axis(tensor, 4, 1, axis: 1), Nx.slice_along_axis(tensor, 5, 80, axis: 1))
  end
  
  defp scale(img) do
    {w, h, _, _}   = CImg.shape(img)
    {wsize, hsize} = @yolo7_shape
    max(w/wsize, h/hsize)
  end
end
