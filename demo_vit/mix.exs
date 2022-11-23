defmodule DemoViT.MixProject do
  use Mix.Project

  def project do
    [
      app: :demo_vit,
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
      mod: {DemoViT.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.4.0"},
      {:cimg, github: "shoz-f/cimg_ex"},
      {:onnx_interp, path: ".."}
    ]
  end
end
