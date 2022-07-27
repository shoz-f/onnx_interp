defmodule DemoNanoDet do
  def apply_nanodet(img_file) do
    img = CImg.load(img_file)
    
    # NanoDet prediciton
    {:ok, res} = NanoDet.apply(img)
    
    IO.inspect(res)

    # draw result box
    Enum.reduce(res, CImg.builder(img), &draw_object(&2, &1))
    |> CImg.run()
  end
  
  def draw_object(builder, {_name, boxes}) do
    Enum.reduce(boxes, builder, fn [_score|box], img ->
      [x0, y0, x1, y1] = Enum.map(box, &round(&1))
      CImg.draw_rect(img, x0, y0, x1, y1, {255, 0, 0})
    end)
  end
end


defmodule NanoDet do
  use OnnxInterp, model: "./nanodet.onnx", label: "./coco.label"

  @nanodet_shape {416, 416}

  def apply(img) do
    # preprocess
    bin = img
      |> CImg.resize(@nanodet_shape)
      |> CImg.to_binary([{:range, {-2.2, 2.7}}, :nchw, :bgr])

    #*+DEBUG:shoz:22/07/24:
    #%Npy{shape: {1,3,416,416}, descr: "<f4", data: bin}
    #|> Npy.save("ex_check/input0.npy")

    # prediction
    outputs = __MODULE__
      |> OnnxInterp.set_input_tensor(0, bin)
      |> OnnxInterp.invoke()
      |> OnnxInterp.get_output_tensor(0)
      |> Nx.from_binary({:f, 32}) |> Nx.reshape({:auto, 112})

    #*+DEBUG:shoz:22/07/24:
    #Nx.reshape(outputs, {1, 3598, 112})
    #|> Npy.save("ex_check/output0.npy")

    # postprocess
    {scores, boxes} =
      Nx.concatenate([outputs, mesh_grid(@nanodet_shape, [8,16,32,64])], axis: 1)
      |> sieve(0.25)
      |> (&{Nx.slice_along_axis(&1, 0, 80, axis: 1), Nx.slice_along_axis(&1, 80, 35, axis: 1)}).()

    #*+DEBUG:shoz:22/07/24:
    #Npy.savez([scores, boxes], "ex_check/scores_boxes0.npz")
    #Npy.savecsv(boxes, "ex_check/boxes0.csv")

    {width, height, _, _} = CImg.shape(img)
    boxes = decode_boxes(boxes, {width, height})

    OnnxInterp.non_max_suppression_multi_class(__MODULE__,
      Nx.shape(scores), Nx.to_binary(boxes), Nx.to_binary(scores), 0.5, 0.25, 0.0,
      boxrepr: :corner
    )
  end

  @doc """
  Create a list of coordinates for mesh grid points.
  """
  def mesh_grid(shape, pitches, opts \\ [])

  def mesh_grid(shape, pitches, opts) when is_list(pitches) do
    Enum.map(pitches, &mesh_grid(shape, &1, opts))
    |> Nx.concatenate()
  end

  def mesh_grid({w, h}, pitch, opts) when w >= 1 and h >= 1 do
    m = trunc(Float.ceil(h/pitch))
    n = trunc(Float.ceil(w/pitch))

    # grid coodinates list
    grid =
      (for y <- 0..(m-1), x <- 0..(n-1), do: [x, y])
      |> Nx.tensor(type: {:f, 32})
      |> (&if :center in opts, do: Nx.add(&1, 0.5), else: &1).()
      |> Nx.multiply(pitch)

    # pitch list
    pitch =
      Nx.broadcast(pitch, {m*n, 1})
      |> Nx.as_type({:f, 32})

    Nx.concatenate([grid, pitch], axis: 1)
  end

  @doc """
  Take records which has score greater or equal than `min_score`.
  """
  def sieve(tensor, min_score) do
    # 各レコードのscores(先頭80要素)が min_score以上かどうかの判定リストを作る
    judge =
      Nx.slice_along_axis(tensor, 0, 80, axis: 1)
      |> Nx.reduce_max(axes: [1])
      |> Nx.greater_equal(min_score)

    # 条件を満たすレコードの数を求める
    count = Nx.sum(judge) |> Nx.to_number()

    # 条件を満たすレコードのindexリストを作る
    index =
      Nx.argsort(judge, direction: :desc)
      |> Nx.slice_along_axis(0, count)

    # 条件を満たすレコードを集めてTensorを作る
    Nx.take(tensor, index)
  end

  @doc """
  """
  def decode_boxes(tensor, world \\ {}) do
    max_index = Nx.axis_size(tensor, 0) - 1

    (for i <- 0..max_index, do: decode_box(tensor[i], world))
    |> Nx.stack()
  end

  @doc """
  """
  def decode_box(tensor, world \\ {}) do
    grid_x = Nx.to_number(tensor[-3])
    grid_y = Nx.to_number(tensor[-2])
    arm    = Nx.iota({8}) |> Nx.multiply(tensor[-1])  # [0, pitch, 2*pitch, ... 7*pitch]

    # private func: decode probability list to wing.
    wing = fn t ->
      max = Nx.reduce_max(t)

      {weight, sum} =
        Nx.subtract(t, max)  # prevent exp from becoming too big
        |> Nx.exp()
        |> (&{&1, Nx.sum(&1)}).()

      Nx.dot(weight, arm) |> Nx.divide(sum) |> Nx.to_number()  # mean of probability list
    end

    {scale_w, scale_h} = scale(world)

    [
      scale_w*(grid_x - wing.(tensor[0..7])),
      scale_h*(grid_y - wing.(tensor[8..15])),
      scale_w*(grid_x + wing.(tensor[16..23])),
      scale_h*(grid_y + wing.(tensor[24..31]))
    ]
    |> keep_within(world)
    |> Nx.stack()
  end

  defp keep_within(box, {}), do: box
  defp keep_within([left,top,right,bottom], {width, height}) do
    [max(left, 0), max(top, 0), min(right, width), min(bottom, height)]
  end
  
  defp scale({}), do: {1.0, 1.0}
  defp scale({width, height}), do: {width/elem(@nanodet_shape, 0), height/elem(@nanodet_shape, 1)}
end
