defmodule OnnxInterp do
  @moduledoc """
  Onnx runtime intepreter for Elixir.
  Deep Learning inference framework.

  ## Basic Usage
  You get the trained onnx model and save it in a directory that your application can read.
  "your-app/priv" may be good choice.

  ```
  $ cp your-trained-model.onnx ./priv
  ```

  Next, you will create a module that interfaces with the deep learning model.
  The module will need pre-processing and post-processing in addition to inference
  processing, as in the example following. OnnxInterp provides inference processing
  only.

  You put `use OnnxInterp` at the beginning of your module, specify the model path as an optional argument. In the inference
  section, you will put data input to the model (`OnnxInterp.set_input_tensor/3`), inference execution (`OnnxInterp.invoke/1`),
  and inference result retrieval (`OnnxInterp.get_output_tensor/2`).

  ```elixr:your_model.ex
  defmodule YourApp.YourModel do
    use OnnxInterp, model: "priv/your-trained-model.onnx"

    def predict(data) do
      # preprocess
      #  to convert the data to be inferred to the input format of the model.
      input_bin = convert-float32-binaries(data)

      # inference
      #  typical I/O data for Onnx models is a serialized 32-bit float tensor.
      output_bin =
        __MODULE__
        |> OnnxInterp.set_input_tensor(0, input_bin)
        |> OnnxInterp.invoke()
        |> OnnxInterp.get_output_tensor(0)

      # postprocess
      #  add your post-processing here.
      #  you may need to reshape output_bin to tensor at first.
      tensor = output_bin
        |> Nx.from_binary({:f, 32})
        |> Nx.reshape({size-x, size-y, :auto})

      * your-postprocessing *
      ...
    end
  end
  ```
  """

  @timeout 300000
  @padding 0

  defmacro __using__(opts) do
    quote generated: true, location: :keep do
      use GenServer

      def start_link(opts) do
        GenServer.start_link(__MODULE__, opts, name: __MODULE__)
      end

      def init(opts) do
        executable = Application.app_dir(:onnx_interp, "priv/onnx_interp")
        opts = Keyword.merge(unquote(opts), opts)
        onnx_model = Keyword.get(opts, :model)
        onnx_label = Keyword.get(opts, :label, "none")
        onnx_opts  = Keyword.get(opts, :opts, "")

        port = Port.open({:spawn_executable, executable}, [
          {:args, String.split(onnx_opts) ++ [onnx_model, onnx_label]},
          {:packet, 4},
          :binary
        ])

        {:ok, %{port: port}}
      end

      def session() do
        %OnnxInterp{module: __MODULE__}
      end

      def handle_call(cmd_line, _from, state) do
        Port.command(state.port, cmd_line)
        response = receive do
          {_, {:data, <<result::binary>>}} -> {:ok, result}
        after
          Keyword.get(unquote(opts), :timeout, 300000) -> {:timeout}
        end
        {:reply, response, state}
      end

      def terminate(_reason, state) do
        Port.close(state.port)
      end
    end
  end

  defstruct module: nil, input: [], output: []

  @doc """
  Get the propaty of the tflite model.

  ## Parameters

    * mod - modules' names
  """
  def info(mod) do
    cmd = 0
    case GenServer.call(mod, <<cmd::little-integer-32>>, @timeout) do
      {:ok, result} ->  Poison.decode(result)
      any -> any
    end
  end

  @doc """
  Stop the tflite interpreter.

  ## Parameters

    * mod - modules' names
  """
  def stop(mod) do
    GenServer.stop(mod)
  end

  @doc """
  Put a flat binary to the input tensor on the interpreter.

  ## Parameters

    * mod   - modules' names or session.
    * index - index of input tensor in the model
    * bin   - input data - flat binary, cf. serialized tensor
    * opts  - data conversion
  """
  def set_input_tensor(mod, index, bin, opts \\ [])

  def set_input_tensor(mod, index, bin, opts) when is_atom(mod) do
    cmd = 1
    case GenServer.call(mod, <<cmd::little-integer-32>> <> input_tensor(index, bin, opts), @timeout) do
      {:ok, result} ->  Poison.decode(result)
      any -> any
    end
    mod
  end

  def set_input_tensor(%OnnxInterp{input: input}=session, index, bin, opts) do
    %OnnxInterp{session | input: [input_tensor(index, bin, opts) | input]}
  end

  defp input_tensor(index, bin, opts) do
    dtype = case Keyword.get(opts, :dtype, "none") do
      "none" -> 0
      "<f4"  -> 1
      "<f2"  -> 2
    end
    {lo, hi} = Keyword.get(opts, :range, {0.0, 1.0})

    size = 16 + byte_size(bin)

    <<size::little-integer-32, index::little-integer-32, dtype::little-integer-32, lo::little-float-32, hi::little-float-32, bin::binary>>
  end

  @doc """
  Invoke prediction.

  ## Parameters

    * mod - modules' names
  """
  def invoke(mod) when is_atom(mod) do
    cmd = 2
    case GenServer.call(mod, <<cmd::little-integer-32>>, @timeout) do
      {:ok, result} -> Poison.decode(result)
      any -> any
    end
    mod
  end

  @doc """
  Get the flat binary from the output tensor on the interpreter.

  ## Parameters

    * mod   - modules' names or session.
    * index - index of output tensor in the model
  """
  def get_output_tensor(mod, index) when is_atom(mod) do
    cmd = 3
    case GenServer.call(mod, <<cmd::little-integer-32, index::little-integer-32>>, @timeout) do
      {:ok, result} -> result
      any -> any
    end
  end

  def get_output_tensor(%OnnxInterp{output: output}, index) do
    Enum.at(output, index)
  end

  @doc """
  Execute the inference session. In session mode, data input/execution of
  inference/output of results to the DL model is done all at once.

  ## Parameters

    * session - session.

  ## Examples.

    ```elixir
      output_bin =
        session()
        |> OnnxInterp.set_input_tensor(0, input_bin)
        |> OnnxInterp.run()
        |> OnnxInterp.get_output_tensor(0)
    ```
  """
  def run(%OnnxInterp{module: mod, input: input}=session) do
    cmd   = 4
    count = Enum.count(input)
    data  = Enum.reduce(input, <<>>, fn x,acc -> acc <> x end)
    case GenServer.call(mod, <<cmd::little-integer-32, count::little-integer-32>> <> data, @timeout) do
      {:ok, <<count::little-integer-32, results::binary>>} ->
          if count > 0 do
              %OnnxInterp{session | output: for <<size::little-integer-32, tensor::binary-size(size) <- results>> do tensor end}
          else
              "error: %{count}"
          end
      any -> any
    end
  end

  @doc """
  Execute post processing: nms.

  ## Parameters

    * mod             - modules' names
    * num_boxes       - number of candidate boxes
    * num_class       - number of category class
    * boxes           - binaries, serialized boxes tensor[`num_boxes`][4]; dtype: float32
    * scores          - binaries, serialized score tensor[`num_boxes`][`num_class`]; dtype: float32
    * iou_threshold   - IOU threshold
    * score_threshold - score cutoff threshold
    * sigma           - soft IOU parameter
  """

  def non_max_suppression_multi_class(mod, {num_boxes, num_class}, boxes, scores, iou_threshold \\ 0.5, score_threshold \\ 0.25, sigma \\ 0.0, opts \\ []) do
    box_repr = case Keyword.get(opts, :boxrepr, :center) do
      :center  -> 0
      :topleft -> 1
      :corner  -> 2
    end

    cmd = 5
    case GenServer.call(mod, <<cmd::little-integer-32, num_boxes::little-integer-32, box_repr::little-integer-32, num_class::little-integer-32, iou_threshold::little-float-32, score_threshold::little-float-32, sigma::little-float-32>> <> boxes <> scores, @timeout) do
      {:ok, nil} -> :notfind
      {:ok, result} -> Poison.decode(result)
      any -> any
    end
  end
end
