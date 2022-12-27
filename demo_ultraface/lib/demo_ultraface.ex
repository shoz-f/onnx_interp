defmodule DemoUltraFace do
  def run() do
    img = CImg.load("./10.jpg")

    with {:ok, res} = UltraFace.apply(img) do
      res["0"]
      |> Enum.take(8)
      |> draw_item(CImg.builder(img), {255,255,0})
      |> CImg.save("result.jpg")
    end
  end

  defp draw_item(boxes, canvas, color \\ {255,255,255}) do
    Enum.reduce(boxes, canvas, fn [_score, x1, y1, x2, y2, _index], canvas ->
      [x1, y1, x2, y2] = PostDNN.clamp([x1, y1, x2, y2], {0.0, 1.0})

      CImg.fill_rect(canvas, x1, y1, x2, y2, color, 0.3)
    end)
  end
end
