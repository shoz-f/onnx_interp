defmodule CandyDemo.MixProject do
  use Mix.Project

  def project do
    [
      app: :candy_demo,
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
      mod: {CandyDemo.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:cimg, github: "shoz-f/cimg_ex"},
      {:onnx_interp, path: "..", env: :test}
    ]
  end
end
