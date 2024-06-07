defmodule OnnxInterp.MixProject do
  use Mix.Project

  def project do
    [
      app: :onnx_interp,
      version: "0.1.11",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      compilers: [:cmake] ++ Mix.compilers(),
      description: description(),
      package: package(),
      deps: deps(),

      cmake: cmake(),

      # Docs
      # name: "onnx_interp",
      source_url: "https://github.com/shoz-f/onnx_interp.git",

      docs: docs()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger, :ssl, :inets]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:castore, "== 0.1.20"},
      {:progress_bar, "~> 2.0"},
      {:mix_cmake, "~> 0.1.4"},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false},
      {:jason, "~> 1.4"}
    ]
  end

  # Cmake configuration.
  defp cmake do
    [
      # Specify cmake build directory or pseudo-path {:local, :global}.
      #   :local(default) - "./_build/.cmake_build"
      #   :global - "~/.#{Cmake.app_name()}"
      #build_dir: :local,

      # Specify cmake source directory.(default: File.cwd!)
      #source_dir: File.cwd!,

      # Specify jobs parallel level.
      build_parallel_level: 4
    ]
    ++ case :os.type do
      {:win32, :nt} -> cmake_win32()
      _ -> []
    end
  end
  
  defp cmake_win32 do
    [
      # Specify generator name.
      # "cmake --help" shows you build-in generators list.
#      generator: "Visual Studio 16 2019",

      # Specify CPU architecture
      platform: "x64",

      # Visual C++ configuration
      build_config: "Release"
    ]
  end

  defp description() do
    "Onnx runtime intepreter for Elixir."
  end

  defp package() do
    [
       name: "onnx_interp",
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => "https://github.com/shoz-f/onnx_interp.git"},
      files: ~w(lib mix.exs README* CHANGELOG* LICENSE* CMakeLists.txt src)
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: [
        "README.md",
#        "LICENSE",
        "CHANGELOG.md",

        #Examples
        "demo_candy/candy.livemd",
        "demo_imgclass/image_classify.livemd",
        "demo_nanodet/nanodet.livemd",
        "demo_yolov4/YOLOv4.livemd",
        "demo_retinaface/RetinaFace.livemd",
        "demo_2d106det/FaceAlign.livemd",
        "demo_centerface/CenterFace.livemd",
        "demo_mbert_qa/mbert_qa.livemd",
        "demo_movenet/MoveNet.livemd",
        "demo_ultraface/UltraFace.livemd",
        "demo_yolop/YOLOP.livemd",
        "demo_yolov4/YOLOv4.livemd",
        "demo_yolov8/YOLOv8.livemd",
      ],
      groups_for_extras: [
        "Examples": Path.wildcard("demo_*/*.livemd")
      ],
#      source_ref: "v#{@version}",
#      source_url: @source_url,
#      skip_undefined_reference_warnings_on: ["CHANGELOG.md"]
    ]
  end
end
