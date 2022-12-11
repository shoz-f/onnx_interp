defmodule MBertQA.Tokenizer do
  @moduledoc """
  Mini Tokenizer for Tensorflow lite's mobileBert example.
  """

  @dic1 :word2token1
  @dic2 :word2token2
  @dic3 :token2word

  @doc """
  Load a vocabulary dictionary.
  """
  def load_dic(fname) do
    Enum.each([@dic1, @dic2, @dic3], fn name ->
      if :ets.whereis(name) != :undefined, do: :ets.delete(name)
      :ets.new(name, [:named_table])
    end)

    File.stream!(fname)
    |> Stream.map(&String.trim_trailing/1)
    |> Stream.with_index()
    |> Enum.each(fn {word, index} ->
      # init dic for encoding.
      case word do
        <<"##", trailing::binary>> -> :ets.insert(@dic2, {trailing, index})
        _ -> :ets.insert(@dic1, {word, index})
      end

      # init dic for decoding.
      :ets.insert(@dic3, {index, word})
    end)
  end

  @doc """
  Tokenize text.
  """
  def tokenize(text, tolower \\ true) do
    if tolower do String.downcase(text) else text end
    |> text2words()
    |> words2tokens()
  end

  defp text2words(text) do
    text
    # cleansing
    |> String.replace(~r/([[:cntrl:]]|\xff\xfd)/, "")
    # separate panc with whitespace
    |> String.replace(~r/([[:punct:]])/, " \\1 ")
    # split with whitespace
    |> String.split()
  end

  defp words2tokens(words) do
    Enum.reduce(words, [], fn word, tokens ->
      case wordpiece1(word, len = String.length(word)) do
        {^len, token} ->
          [token | tokens]

        {0, token} ->
          [token | tokens]

        {n, token} ->
          len = len - n
          wordpiece2(String.slice(word, n, len), len, [token | tokens])
      end
    end)
    |> Enum.reverse()
  end

  defp wordpiece1(_, 0), do: {0, token("[UNK]")}

  defp wordpiece1(word, n) do
    case lookup_1(String.slice(word, 0, n)) do
      {_piece, token} ->
        {n, token}

      nil ->
        wordpiece1(word, n - 1)
    end
  end

  defp wordpiece2(_, 0, tokens), do: [token("[UNK]") | tokens]

  defp wordpiece2(word, n, tokens) do
    case lookup_2(String.slice(word, 0, n)) do
      {_piece, token} ->
        len = String.length(word) - n

        if len == 0 do
          [token | tokens]
        else
          wordpiece2(String.slice(word, n, len), len, [token | tokens])
        end

      nil ->
        wordpiece2(word, n - 1, tokens)
    end
  end

  defp lookup_1(x), do: lookup(@dic1, x)
  defp lookup_2(x), do: lookup(@dic2, x)
  defp lookup(dic, x), do: List.first(:ets.lookup(dic, x))

  @doc """
  Get token of `word`.
  """
  def token(word) do
    try do
      :ets.lookup_element(@dic1, word, 2)
    rescue
      ArgumentError -> nil
    end
  end

  @doc """
  Decode the token list `tokens`.
  """
  def decode(tokens) do
    Enum.map(tokens, fn x -> :ets.lookup_element(@dic3, x, 2) end)
  end
end
