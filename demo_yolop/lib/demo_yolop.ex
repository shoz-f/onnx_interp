defmodule DemoYOLOP do
  def run(path) do
    img = CImg.load(path)

    with {:ok, cars, drive_area, lane_line} = YOLOP.apply(img) do
      CImg.builder(img)
      |> CImg.paint_mask(drive_area, {255,255,0})
      |> CImg.paint_mask(lane_line,  {255,0,0})
      |> draw_item(cars, {0,192,0})
      |> CImg.save("result.jpg")
    end
  end

  defp draw_item(canvas, bbox, color) do
    Enum.reduce(bbox, canvas, fn [_score, x1, y1, x2, y2, _index], canvas ->
      CImg.fill_rect(canvas, x1, y1, x2, y2, color, 0.3)
    end)
  end
end
