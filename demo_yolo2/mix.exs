defmodule DemoYolo2.MixProject do
  use Mix.Project

  def project do
    [
      app: :demo_yolo2,
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
      mod: {DemoYolo2.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:cimg, github: "shoz-f/cimg_ex"},
      {:onnx_interp, path: "..", env: :test},
      {:nx, "~> 0.2.1"},
      {:postdnn, "~> 0.1.2"},
      {:npy, github: "shoz-f/npy_ex"}
    ]
  end
end
