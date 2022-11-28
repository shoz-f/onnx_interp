defmodule DemoCandy do
  def run(filename \\ "flog.jpg") do
    CImg.load(filename)
    |> Candy.apply()
    |> CImg.save("candy.jpg")
  end
end
