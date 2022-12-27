defmodule CenterFace do
  import Nx.Defn

  @width  640
  @height 640

  alias OnnxInterp, as: NNInterp
  use NNInterp,
    model: "./model/centerface_dynamic.onnx",
    inputs: [f32: {1,3,@height,@width}],
    outputs: [f32: {1,1,div(@height,4),div(@width,4)}, f32: {1,2,div(@height,4),div(@width,4)}, f32: {1,2,div(@height,4),div(@width,4)}, f32: {1,10,div(@height,4),div(@width,4)}]

  def apply(img) do
    # preprocess
    bin = CImg.builder(img)
      |> CImg.resize({@width, @height}, :ul, 0)
      |> CImg.to_binary([{:range, {0.0, 255.0}}, :nchw])

    # prediction
    outputs = session()
      |> NNInterp.set_input_tensor(0, bin)
      |> NNInterp.invoke()

    [heatmap, scale, offset, landm] = Enum.with_index([1, 2, 2, 10], fn dim,i ->
        NNInterp.get_output_tensor(outputs, i) |> Nx.from_binary(:f32) |> Nx.reshape({dim, :auto})
      end)

    # postprocess
    scores = Nx.transpose(heatmap)
    boxes  = decode_boxes(offset, scale)

    {:ok, res} = NNInterp.non_max_suppression_multi_class(__MODULE__,
        Nx.shape(scores), Nx.to_binary(boxes), Nx.to_binary(scores),
        iou_threshold: 0.2, score_threshold: 0.2,
        boxrepr: :corner)

    {:ok, fit2image_with_landmark(landm, res["0"], inv_aspect(img))}
  end


  @grid PostDNN.meshgrid({@width, @height}, [4], [:center, :normalize, :transpose])

  defp decode_boxes(offset, size) do
    # decode box center coordinate on {1.0, 1.0}
    center = offset
      |> Nx.reverse(axes: [0])     # swap (y,x) -> (x,y)
      |> Nx.multiply(@grid[2..3]) # * grid_pitch(x,y)
      |> Nx.add(@grid[0..1])      # + grid(x,y)

    # decode box half size
    half_size = size
      |> Nx.reverse(axes: [0])     # swap (y,x) -> (x,y)
      |> Nx.exp()
      |> Nx.multiply(@grid[2..3]) # * grid_pitch(x,y)
      |> Nx.divide(2.0)

    # decode boxes
    [Nx.subtract(center, half_size), Nx.add(center, half_size)]
      |> Nx.concatenate()
      |> PostDNN.clamp({0.0, 1.0})
      |> Nx.transpose()
  end

  defp fit2image_with_landmark(landm, nms_res, {inv_x, inv_y} \\ {1.0, 1.0}) do
    Enum.map(nms_res, fn [score, x1, y1, x2, y2, index] ->
#      priorbox = Nx.slice_along_axis(@grid, index, 1, axis: 1) |> Nx.squeeze()
#
#      landmark = landm[index]
#        |> Nx.reshape({:auto, 2})
#        |> Nx.multiply(priorbox[2..3]) # * prior_size(x,y)
#        |> Nx.add(priorbox[0..1])      # + grid(x,y)
#        |> Nx.multiply(Nx.tensor([inv_x, inv_y]))
#        |> Nx.to_flat_list()
#        |> Enum.chunk_every(2)
#
      [score, x1*inv_x, y1*inv_y, x2*inv_x, y2*inv_y, index]
    end)
  end

  defp inv_aspect(img) do
    {w, h, _, _} = CImg.shape(img)
    if w > h, do: {1.0, w / h}, else: {h / w, 1.0}
  end
end
