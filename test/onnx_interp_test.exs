defmodule OnnxInterpTest do
  use ExUnit.Case
  doctest OnnxInterp

  defmodule Foo do
    use OnnxInterp, model: "priv/candy.onnx"
  end

  test "greets the world" do
    Foo.start_link([])
    OnnxInterp.info(Foo)
    |> IO.inspect
  end
end
