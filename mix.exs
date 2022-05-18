defmodule OnnxInterp.MixProject do
  use Mix.Project

  def project do
    [
      app: :onnx_interp,
      version: "0.1.0",
      elixir: "~> 1.12",
      start_permanent: Mix.env() == :prod,
      compilers: [:cmake] ++ Mix.compilers(),
      deps: deps(),

      cmake: cmake(),
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:poison, "~> 3.1"},
      {:mix_cmake, github: "shoz-f/mix_cmake"},
      {:cimg, github: "shoz-f/cimg_ex"},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
    ]
  end

  # Cmake configuration.
  defp cmake do
    [
      # Specify cmake build directory or pseudo-path {:local, :global}.
      #   :local(default) - "./_build/.cmake_build"
      #   :global - "~/.#{Cmake.app_name()}"
      #
      #build_dir: :local,

      # Specify cmake source directory.(default: File.cwd!)
      #
      #source_dir: File.cwd!,

      # Specify generator name.
      # "cmake --help" shows you build-in generators list.
      #
      generator: "Visual Studio 16 2019",

      # Specify jobs parallel level.
      #
      build_parallel_level: 4,
      
      # Specify CPU architecture
      platform: "x64",
      
      # Visual C++ configuration
      build_config: "Debug"
    ]
  end

end
