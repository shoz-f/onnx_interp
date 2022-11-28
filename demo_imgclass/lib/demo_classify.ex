defmodule DemoClassify do
  def run(filename \\ "lion.jpg", top \\ 3) do
    CImg.load(filename)
    |> ImageClassify.apply(top)
    |> tap(fn _ -> IO.puts("#{ImageClassify.info.name} answers:") end)
    |> IO.inspect(label: "'#{filename}' is ")
    :ok
  end
end
