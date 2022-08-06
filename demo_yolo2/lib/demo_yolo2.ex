defmodule DemoYolo2 do
  use OnnxInterp, model: "./yolov2-coco-9.onnx", label: "./coco.label"

  @label         (for item <- File.stream!("./coco.label") do String.trim_trailing(item) end)
                  |> Enum.with_index(&{&2, &1})
                  |> Enum.into(%{})

  @yolo2_input  {416, 416}
  @yolo2_output {5, 85, 13, 13}
  @grid          PostDNN.meshgrid(@yolo2_input, 32, [:transpose]) |> Nx.tile([5])
  @anchors       Nx.tensor([[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]])

  def apply(img) do
    # preprocess
    bin = img
      |> CImg.resize(@yolo2_input)
      |> CImg.to_binary([:nchw])

    # prediction
    outputs = __MODULE__
      |> OnnxInterp.set_input_tensor(0, bin)
      |> OnnxInterp.invoke()
      |> OnnxInterp.get_output_tensor(0)
      |> Nx.from_binary({:f, 32}) |> Nx.reshape(@yolo2_output)

    # postprocess
    outputs = Nx.transpose(outputs, axes: [1, 0, 2, 3]) |> Nx.reshape({85, :auto})
      # outputs: [box(4),box_score(1),class_score(80)]x[anchor0[13x13],anchor1[13x13],..,anchor4[13x13]]

    boxes  = decode_boxes(outputs)
    scores = decode_scores(outputs)

    PostDNN.non_max_suppression_multi_class(
      Nx.shape(scores), Nx.to_binary(boxes), Nx.to_binary(scores), label: @label
    )
  end

  def decode_boxes(t) do
    # decode box center coordinate on @yolo2_input
    center = Nx.logistic(t[0..1])
      |> Nx.multiply(@grid[2])  # * pitch
      |> Nx.add(@grid[0..1])    # + grid(x,y)
      |> Nx.transpose()

    # decode box size
    size = Nx.exp(t[2..3])
      |> Nx.multiply(@grid[2]) # * pitch
      # multiply @anchors
      |> Nx.reshape({2, 5, :auto})
      |> Nx.transpose(axes: [2, 1, 0])
      |> Nx.multiply(@anchors)
      # get a transposed box sizes.
      |> Nx.transpose(axes: [1, 0, 2])
      |> Nx.reshape({:auto, 2})

    Nx.concatenate([center, size], axis: 1)
  end

  def decode_scores(t) do
    # decode box confidence
    confidence = Nx.logistic(t[4])

    # decode class scores: (softmax normalized class score)*(box confidence)
    exp = Nx.exp(t[5..-1//1])

    Nx.divide(exp, Nx.sum(exp, axes: [0])) # apply softmax on each class score
    |> Nx.multiply(confidence)
    |> Nx.transpose()
  end
  
  def coco_label() do
    @label
  end
end
