defmodule DemoYOLOv8 do
  @palette CImg.Util.rand_palette("./model/coco.label")

  def run(path) when is_binary(path) do
    run(YOLOv8, CImg.load(path))
  end

  def run(yolo, %CImg{}=img) do
    with {:ok, res} <- yolo.apply(img) do
      Enum.reduce(res, CImg.builder(img), &draw_item(&1, &2))
      |> CImg.save("#{yolo}.jpg")
    end
  end

  defp draw_item({name, boxes}, canvas) do
    color = @palette[name]
    Enum.reduce(boxes, canvas, fn [_score, x1, y1, x2, y2, _index], canvas ->
      [x1,y1,x2,y2] = PostDNN.clamp([x1,y1,x2,y2], {0.0, 1.0})

      CImg.fill_rect(canvas, x1, y1, x2, y2, color, 0.4)
    end)
  end
end
