defmodule DemoYOLOP do
  def run(path) do
    img = CImg.load(path)

    with {:ok, bbox, mask} = YOLOP.apply(img) do
      mask = CImg.color_mapping(mask, [{0,0,0},{255,0,128},{255,0,0}])

      CImg.builder(img)
      |> CImg.blend(mask, 0.3)
      |> draw_item(bbox, {0,192,0})
      |> CImg.save("result.jpg")
    end
  end

  defp draw_item(canvas, bbox, color) do
    Enum.reduce(bbox, canvas, fn [_score, x1, y1, x2, y2, _index], canvas ->
      CImg.fill_rect(canvas, x1, y1, x2, y2, color, 0.3)
    end)
  end
end
