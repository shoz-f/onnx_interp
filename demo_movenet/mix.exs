defmodule DemoMovenet.MixProject do
  use Mix.Project

  def project do
    [
      app: :demo_movenet,
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
      mod: {DemoMovenet.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:onnx_interp, path: ".."},
      {:nx, "~> 0.4.0"},
      {:cimg, "~> 0.1.16"},
      {:postdnn, "~> 0.1.4"},
      {:exprintf, "~> 0.2.1"}
    ]
  end
end
