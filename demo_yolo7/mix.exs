defmodule DemoYolo7.MixProject do
  use Mix.Project

  def project do
    [
      app: :demo_yolo7,
      version: "0.1.0",
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {DemoYolo7.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:cimg, github: "shoz-f/cimg_ex"},
      {:onnx_interp, path: "..", env: :test},
      {:nx, "~> 0.4.0"}
    ]
  end
end
