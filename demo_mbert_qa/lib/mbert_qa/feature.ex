defmodule MBertQA.Feature do
  @moduledoc """
  Feature converter for Tensorflow lite's mobileBert example.

  Feature tensor[4][] is consisted of
    [0] list of tokens converted from query and contex text in the dictionary.
    [1] list of token masks. all are 1 in valid token.
    [2] segment type indicator. 0 for query, 1 for context.
    [3] index of the token's position in the context list. see bellow.

  Context list is a list of words separated by whitespace from the context .
  """

  alias MBertQA.Tokenizer

  @max_seq   384
  @max_query 64

  defdelegate load_dic(fname), to: Tokenizer, as: :load_dic

  @doc """
  """
  def convert(query, context) do
    {t_query, room} = convert_query(query)
    {t_context, room, context_list} = convert_context(context, room)

    {
      Nx.concatenate([cls(), t_query, sep(0), t_context, sep(1), padding(room)], axis: 1),
      context_list
    }
  end

  defp convert_query(text, room \\ @max_seq-3) do
    t_query = convert_text(text, @max_query, 0)

    {
      t_query,
      room - Nx.axis_size(t_query, 1)
    }
  end

  defp convert_context(text, room) do
    context_list = String.split(text)

    {t_context, room} = context_list
      |> Stream.with_index()
      |> Enum.reduce_while({[], room}, fn
        {_, _},{_, 0}=res ->
          {:halt, res}
      
        {word, index},{t_context, max_len} ->
          t_word = convert_text(word, max_len, 1, index)
          {:cont, {[t_word|t_context], max_len - Nx.axis_size(t_word, 1)}}
      end)

    {
      Enum.reverse(t_context) |> Nx.concatenate(axis: 1),
      room,
      context_list
    }
  end

  defp convert_text(text, max_len, segment, index \\ -1) do
    tokens = text
      |> Tokenizer.tokenize()
      |> Enum.take(max_len)
      |> Nx.tensor()

    len = Nx.axis_size(tokens, 0)

    Nx.stack([tokens, Nx.broadcast(1, {len}), Nx.broadcast(segment, {len}), Nx.broadcast(index, {len})])
    |> Nx.as_type({:s, 32})
  end

  defp cls() do
    Nx.tensor([[Tokenizer.token("[CLS]")], [1], [0], [-1]], type: {:s, 32})
  end

  defp sep(segment) do
    Nx.tensor([[Tokenizer.token("[SEP]")], [1], [segment], [-1]], type: {:s, 32})
  end

  defp padding(n) do
    Nx.tensor([0, 0, 0, -1], type: {:s, 32})
    |> Nx.broadcast({4, n}, axes: [0])
  end
end
