defmodule DemoYolo7.Application do
  # See https://hexdocs.pm/elixir/Application.html
  # for more information on OTP Applications
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # Starts a worker by calling: DemoYolo7.Worker.start_link(arg)
      # {DemoYolo7.Worker, arg}
      DemoYolo7
    ]

    # See https://hexdocs.pm/elixir/Supervisor.html
    # for other strategies and supported options
    opts = [strategy: :one_for_one, name: DemoYolo7.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
