defmodule DemoMBertQA do
  def run() do
    MBertQA.setup()
    
    context = File.read!("passage.txt")
    IO.puts(">CONTEXT:\n#{context}")

    query = "Who is the CEO of Google?"
    IO.puts(">QUESTION:\n#{query}")

    MBertQA.apply(query, context)
    |> Enum.each(fn {ans, score} ->
      IO.puts("\n>ANS: \"#{ans}\", score:#{score}")
    end)
  end
end
