defmodule DemoRetinaFace do
  def run() do
    img = CImg.load("./img2.jpg")

    with {:ok, res} = RetinaFace.apply(img) do
      res
      |> draw_item(CImg.builder(img), {255,0,255})
      |> CImg.save("result.jpg")
    end
  end

  defp draw_item(boxes, canvas, color \\ {255,255,255}) do
    Enum.reduce(boxes, canvas, fn [_score, x1, y1, x2, y2, _landmark], canvas ->
      CImg.fill_rect(canvas, x1, y1, x2, y2, color, 0.3)
    end)
  end
end
