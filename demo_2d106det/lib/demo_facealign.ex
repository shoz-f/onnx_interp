defmodule DemoFaceAlign do
  def run(path) do
    img = CImg.load(path)

    with {:ok, res} = FaceAlign.apply(img) do
      res
      |> draw_item(CImg.builder(img), {0,255,0})
      |> CImg.save("result.jpg")
    end
  end

  defp draw_item(landmark, canvas, color \\ {255,255,255}) do
    Enum.reduce(landmark, canvas, fn [x, y], canvas ->
      CImg.draw_marker(canvas, x, y, :red)
    end)
  end
end
