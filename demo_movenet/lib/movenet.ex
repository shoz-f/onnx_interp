defmodule Movenet do

  @width  192
  @height 192

  alias OnnxInterp, as: NNInterp
  use NNInterp,
    model: "./model/movenet_singlepose.onnx",
    url: "https://github.com/shoz-f/onnx_interp/releases/download/models/movenet_singlepose.onnx",
    inputs: [i32: {1,@height,@width,3}],
    outputs: [f32: {1,1,17,3}]

  def apply(img) do
    # preprocess
    input0 = CImg.builder(img)
      |> CImg.resize({@width, @height}, :ul, 0)
      |> CImg.to_binary(dtype: "<i4")
    
    # prediction
    output0 = session()
      |> NNInterp.set_input_tensor(0, input0)
      |> NNInterp.invoke()
      |> NNInterp.get_output_tensor(0)
      |> Nx.from_binary(:f32) |> Nx.reshape({17, 3})
    
    # postprocess
    {inv_w, inv_h} = inv_aspect(img)
    joints = Nx.multiply(output0, Nx.tensor([inv_h, inv_w, 1.0]))

    {:ok, to_bones(joints)}
  end

  @bones [
    { 0,  1, :fuchsia},
    { 0,  2, :aqua   },
    { 1,  3, :fuchsia},
    { 2,  4, :aqua   },
    { 0,  5, :fuchsia},
    { 0,  6, :aqua   },
    { 5,  7, :fuchsia},
    { 7,  9, :fuchsia},
    { 6,  8, :aqua   },
    { 8, 10, :aqua   },
    { 5,  6, :yellow },
    { 5, 11, :fuchsia},
    { 6, 12, :aqua   },
    {11, 12, :yellow },
    {11, 13, :fuchsia},
    {13, 15, :fuchsia},
    {12, 14, :aqua   },
    {14, 16, :aqua   },
  ]
  
  def to_bones(t, threshold \\ 0.11) do
    Enum.flat_map(@bones, fn {p1, p2, color} ->
      [y1,x1,score1] = Nx.to_flat_list(t[p1])
      [y2,x2,score2] = Nx.to_flat_list(t[p2])

      if (score1 > threshold) && (score2 > threshold) do
        [{x1, y1, x2, y2, color}]
      else
        []
      end
    end)
  end

  defp inv_aspect(img) do
    {w, h, _, _} = CImg.shape(img)
    if w > h, do: {1.0, w/h}, else: {h/w, 1.0}
  end
end
