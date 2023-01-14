defmodule YOLOv8 do
  @moduledoc """
  Original work:
    Ultralytics YOLOv8 - https://github.com/ultralytics/ultralytics
  """

  @width  640
  @height 640
  
  alias OnnxInterp, as: NNInterp
  use NNInterp, label: "./model/coco.label",
    model: "./model/yolov8n.onnx",
    url: "https://github.com/shoz-f/onnx_interp/releases/download/models/yolov8n.onnx",
    inputs: [f32: {1,3,@width,@height}],
    outputs: [f32: {1,84,8400}]

  def apply(img) do
    # preprocess
    input0 = CImg.builder(img)
      |> CImg.resize({@width, @height})
      |> CImg.to_binary([{:range, {0.0, 1.0}}, :nchw])

    # prediction
    output0 = session()
      |> NNInterp.set_input_tensor(0, input0)
      |> NNInterp.invoke()
      |> NNInterp.get_output_tensor(0)
      |> Nx.from_binary(:f32) |> Nx.reshape({84,8400})

    # postprocess
    scores = Nx.transpose(output0[4..-1//1])
    boxes  = Nx.transpose(output0[0..3])

    NNInterp.non_max_suppression_multi_class(__MODULE__,
      Nx.shape(scores), Nx.to_binary(boxes), Nx.to_binary(scores)
    )
    |> ratio_box()
  end

  defp ratio_box({:ok, result}) do
    clamp = fn x -> min(max(x, 0.0), 1.0) end
    {
      :ok,
      Enum.reduce(Map.keys(result), result, fn key,map ->
        Map.update!(map, key, &Enum.map(&1, fn [score, x1, y1, x2, y2, index] ->
          [score, clamp.(x1/@width), clamp.(y1/@height), clamp.(x2/@width), clamp.(y2/@height), index]
        end))
      end)
    }
  end
end
