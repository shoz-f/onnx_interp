defmodule UltraFace.MixProject do
  use Mix.Project

  def project do
    [
      app: :ultra_face,
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
      mod: {UltraFace.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:onnx_interp, path: ".."},
      {:cimg, "~> 0.1.14"},
      {:postdnn, "~> 0.1.4"},
      {:nx, "~> 0.4.0"}
    ]
  end
end
