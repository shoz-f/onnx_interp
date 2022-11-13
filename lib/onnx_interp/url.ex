defmodule OnnxInterp.URL do
  @doc """
  Download and process data from url.
  
  ## Parameters
    * url - download site url
    * func - function to process downloaded data
  """
  def download(url, func) when is_function(func) do
    IO.puts("Downloading \"#{url}\".")

    with {:ok, res} <- HTTPoison.get(url, [], follow_redirect: true) do
      IO.puts("...processing.")
      func.(res.body)
    end
  end

  @doc """
  Download and save the file from url.
  
  ## Parameters
    * url - download site url
    * path - distination path of downloaded file
    * name - name for the downloaded file
  """
  def download(url, path \\ "./", name \\ nil) do
    IO.puts("Downloading \"#{url}\".")

    with {:ok, res} <- HTTPoison.get(url, [], follow_redirect: true),
      {_, <<"attachment; filename=", fname::binary>>} <- List.keyfind(res.headers, "Content-Disposition", 0),
      :ok <- File.mkdir_p(path)
    do
      Path.join(path, name||fname)
      |> save(res.body)
    end
  end

  defp save(file, bin) do
    with :ok <- File.write(file, bin) do
      IO.puts("...finish.")
      {:ok, file}
    end
  end
end
