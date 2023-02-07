defmodule DemoScDepth do
  def run(path) do
    img = CImg.load(path)
    
    ScDepth.apply(img)
  end
end
