defmodule DemoUltraFace do
  def run(path) do
    img = CImg.load(path)

    with {:ok, res} = UltraFace.apply(img) do
      res
      |> draw_item(CImg.builder(img), {0,255,0})
      |> CImg.save("result.jpg")
    end
  end

  defp draw_item(boxes, canvas, color \\ {255,255,255}) do
    Enum.reduce(boxes, canvas, fn [_score, x1, y1, x2, y2, _index], canvas ->
      CImg.fill_rect(canvas, x1, y1, x2, y2, color, 0.4)
    end)
  end
end
