defmodule OnnxInterp.URL do
  @doc """
  Download and process data from url.
  
  ## Parameters
    * url - download site url
    * func - function to process downloaded data
  """
  def download(url, func) when is_function(func) do
    IO.puts("Downloading \"#{url}\".")

    response = get!(url)

    IO.puts("...processing.")
    func.(response.body)
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

    response = get!(url)

    name = name || case attachment_filename(response.headers) do
      {:ok, name} -> name
      _ -> IO.puts("** 'noname.bin' was used due to lack of a valid file name **")
           "noname.bin"
    end

    File.mkdir_p(path)

    Path.join(path, name)
    |> save(response.body)
  end

  defp save(file, bin) do
    with :ok <- File.write(file, bin) do
      IO.puts("...finish.")
      {:ok, file}
    end
  end

  defp get!(url) do
    http_opts = [
      ssl: [
        verify: :verify_peer,
        cacertfile: CAStore.file_path(),
        customize_hostname_check: [
          match_fun: :public_key.pkix_verify_hostname_match_fun(:https)
        ]
      ]
    ]

    case :httpc.request(:get, {url, []}, http_opts, body_format: :binary) do
      {:ok, {{_, status, _}, headers, body}} ->
        if status >= 400 do
          raise "HTTP #{status} #{inspect(body)}"
        else
          %{status: status, headers: headers, body: body}
        end

      {:error, reason} ->
        raise inspect(reason)
    end
  end
  
  defp attachment_filename(headers) do
    with {_, cd} <- List.keyfind(headers, 'content-disposition', 0),
         [[_, fname]] <- Regex.scan(~r/filename="?(.+)"?/, List.to_string(cd))
    do
      {:ok, fname}
    else
      _ -> :none
    end
  end
end
