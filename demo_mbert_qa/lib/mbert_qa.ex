defmodule MBertQA do
  @moduledoc """
  Documentation for `MBertQA`.
  """
  alias OnnxInterp, as: NNInterp
  use NNInterp,
    model: "./model/mobilebert_squad.onnx",
    url: "https://github.com/shoz-f/onnx_interp/releases/download/models/mobilebert_squad.onnx"

  alias MBertQA.Feature

  @max_ans 32
  @predict_num 5

  def setup() do
    Feature.load_dic("./model/vocab.txt")
  end

  def apply(query, context, predict_num \\ @predict_num) do
    # pre-processing
    {feature, context_list} = Feature.convert(query, context)

    # prediction
    mod = __MODULE__
      |> NNInterp.set_input_tensor(0, Nx.to_binary(feature[0]))
      |> NNInterp.set_input_tensor(1, Nx.to_binary(feature[1]))
      |> NNInterp.set_input_tensor(2, Nx.to_binary(feature[2]))
      |> NNInterp.invoke()

    [end_logits, beg_logits] = Enum.map(0..1, fn x ->
      NNInterp.get_output_tensor(mod, x)
      |> Nx.from_binary(:f32)
    end)

    # post-processing
    [beg_index, end_index] = Enum.map([beg_logits, end_logits], fn t ->
      Nx.argsort(t, direction: :desc)
      |> Nx.slice_along_axis(0, predict_num)
      |> Nx.to_flat_list()
      |> Enum.filter(&(feature[3][&1] >= 0))
    end)

    for b <- beg_index, e <- end_index, b <= e, e - b + 1 < @max_ans do
      {b, e, Nx.to_number(Nx.add(beg_logits[b], end_logits[e]))}
    end
    |> Enum.sort(&(elem(&1, 2) >= elem(&2, 2)))
    |> Enum.take(predict_num)
    # make answer text with score
    |> Enum.map(fn {b, e, score} ->
      b = Nx.to_number(feature[3][b])
      e = Nx.to_number(feature[3][e])
      {
        Enum.slice(context_list, b..e) |> Enum.join(" "),
        score
      }
    end)
  end
end
