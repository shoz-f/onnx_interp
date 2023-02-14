defmodule DemoHRDepth.MixProject do
  use Mix.Project

  def project do
    [
      app: :demo_scdepth,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {DemoHRDepth.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:onnx_interp, path: ".."},
      {:cimg, github: "shoz-f/cimg_ex"},
      {:nx, "~> 0.4.0"}
    ]
  end
end
