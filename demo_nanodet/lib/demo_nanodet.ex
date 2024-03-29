defmodule DemoNanoDet do
  @palette CImg.Util.rand_palette("./model/coco.label")

  def run(filename \\ "dog.jpg") do
    img = CImg.load(filename)
    
    # NanoDet prediciton
    {:ok, res} = NanoDet.apply(img)

    # draw result box
    Enum.reduce(res, CImg.builder(img), &draw_item(&1, &2))
    |> CImg.save("nanodet.jpg")
  end
  
  defp draw_item({name, boxes}, canvas) do
    color = @palette[name]
    Enum.reduce(boxes, canvas, fn [_score, x1, y1, x2, y2, _index], img ->
      CImg.fill_rect(img, x1, y1, x2, y2, color, 0.3)
    end)
  end
end
