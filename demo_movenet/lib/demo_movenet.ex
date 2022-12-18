defmodule DemoMovenet do
  def run(src, dst, range) do
    Enum.each(range, fn i ->
      name = ExPrintf.sprintf("%03d.jpg", [i])
      src_file = Path.join(src, name)
      dst_file = Path.join(dst, name)
      run(src_file, dst_file)
    end)
  end

  def run(src, dst \\ "xxx.jpg") do
    img = CImg.load(src)

    with {:ok, res} <- Movenet.apply(img) do
      Enum.reduce(res, CImg.builder(img), &draw_item(&1, &2))
      |> CImg.save(dst)
    end
  end
  
  defp draw_item({x1, y1, x2, y2, color}, canvas) do
    CImg.draw_line(canvas, x1, y1, x2, y2, color, thick: 7)
  end
end
