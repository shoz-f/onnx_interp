defmodule OnnxInterpTest do
  use ExUnit.Case
  doctest OnnxInterp

  defmodule Mnist do
    use OnnxInterp, model: "test/mnist.onnx"
  end

  test "greets the world" do
    Mnist.start_link([])
    OnnxInterp.info(Mnist)
    |> IO.inspect
  end
end
