defmodule DemoHRDepth do
  def run(path) do
    img = CImg.load(path)
    
    HRDepth.apply(img)
  end
end
