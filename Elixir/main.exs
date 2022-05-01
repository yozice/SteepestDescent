defmodule Parallel do
  def pmap(collection, func) do
    collection
    |> Enum.map(&(Task.async(fn -> func.(&1) end)))
    |> Enum.map(fn x -> Task.await(x, 10000) end)
  end
end

defmodule MatrixOps do

  def get_matrix(fname) do
    stream = File.stream!(fname)
    #{ns,_} = Enum.take(stream,1) |> List.first() |> String.replace("\n","") |>Float.parse()
    strings = Enum.take(stream,1000000)
    values = Enum.map(strings,fn line -> line |> String.split |> Enum.map(fn value -> Float.parse(value) |> elem(0) end) end)
    [n|matrix] = values
    [n | _] = n
    n = trunc(n)
    %{n: n,matrix: matrix}
  end

  def get_vector(fname) do
    stream = File.stream!(fname)
    strings = Enum.take(stream,1000000)
    values = Enum.map(strings,fn line -> line |> String.split |> Enum.map(fn value -> Float.parse(value) |> elem(0) end) end)
    [n|vector] = values
    [n | _] = n
    n = trunc(n)
    [vector | _] = vector
    %{n: n,vector: vector}
  end

  def const_multiple(const, x) when is_number(x) do
    const * x
  end

  def const_multiple(const, x) when is_list(x) do
    Enum.map(x, &const_multiple(const, &1))
  end

  #dot product for two vectors
  def scalar_product(a,b) do
    Enum.zip(a,b) |> Enum.map(fn {x,y} -> x*y end) |> Enum.sum()
  end

  def matvect_product(matrix_a, vector_x) do
    Enum.map(matrix_a, fn(row) -> scalar_product(row, vector_x) end)
  end

  def gradF(matrix_a, vector_x, vector_b) do
    matvect_product(matrix_a, vector_x) |> Enum.zip(vector_b) |> Enum.map(fn {y,b} -> b-y end)
  end

  def vector_norm(vec) do
    Enum.map(vec,fn x->x*x end) |> Enum.sum() |> :math.sqrt()
  end
end

defmodule MatrixOpsParallel do

  def get_matrix(fname) do
    stream = File.stream!(fname)
    #{ns,_} = Enum.take(stream,1) |> List.first() |> String.replace("\n","") |>Float.parse()
    strings = Enum.take(stream,1000000)
    values = Enum.map(strings,fn line -> line |> String.split |> Enum.map(fn value -> Float.parse(value) |> elem(0) end) end)
    [n|matrix] = values
    [n | _] = n
    n = trunc(n)
    %{n: n,matrix: matrix}
  end

  def get_vector(fname) do
    stream = File.stream!(fname)
    strings = Enum.take(stream,1000000)
    values = Enum.map(strings,fn line -> line |> String.split |> Enum.map(fn value -> Float.parse(value) |> elem(0) end) end)
    [n|vector] = values
    [n | _] = n
    n = trunc(n)
    [vector | _] = vector
    %{n: n,vector: vector}
  end

  def const_multiple(const,x,chunksize) do
    # x
    # |> Parallel.pmap(fn x -> x*const end)
    # |> Enum.flat_map(fn x -> x end)
    x
    |> Enum.chunk_every(chunksize)
    |> Parallel.pmap(fn x -> MatrixOps.const_multiple(const,x) end)
    |> Enum.flat_map(fn x -> x end)

    # Enum.map(x, fn x -> x*const end)
  end

  def mult(a,b) do
    a*b
  end

  # def mult(a,b) when is_list(a) do
  #   IO.inspect(a)
  #   IEx.Info.info(a)
  #   |> IO.inspect
  #   Enum.zip(a,b) |> Enum.map(fn {x,y} -> x*y end)
  # end

  def mult_chunk_xy([[]|_]), do: []

  def mult_chunk_xy([{x,y}|[]]) do
    [mult(x,y)]
  end

  def mult_chunk_xy([{x,y}|tail]) do
    [mult(x,y)]++mult_chunk_xy(tail)
  end

  #dot product for two vectors
  def scalar_product(a,b, chunk_size) do
    Enum.zip(a,b)
    |> Enum.chunk_every(chunk_size)
    |> Parallel.pmap(fn chunk -> mult_chunk_xy(chunk) end)
    |> Enum.flat_map(fn x -> x end)
    |> Enum.sum()
  end

  def matvect_product(matrix_a, vector_x, chunk_size) do
    matrix_a
    # |> Enum.chunk_every(chunk_size)
    |> Enum.map(fn(row) -> scalar_product(row, vector_x, chunk_size) end)
    # |> Enum.flat_map(fn x -> x end)
  end

  def add_rows(a,b) do
    a+b
  end

  def add_all_rows([[]|_]), do: []
  def add_all_rows([{v1,v2}|[]]) do
    [add_rows(v1,v2)]
  end
  def add_all_rows([{v1,v2}|tail]) do
    [add_rows(v1,v2)]++add_all_rows(tail)
  end

  def substract_rows(a,b) do
    b-a
  end

  def substract_all_rows([[]|_]), do: []
  def substract_all_rows([{v1,v2}|[]]) do
    [substract_rows(v1,v2)]
  end
  def substract_all_rows([{v1,v2}|tail]) do
    [substract_rows(v1,v2)]++substract_all_rows(tail)
  end

  def gradF(matrix_a, vector_x, vector_b, chunk_size) do
    MatrixOps.matvect_product(matrix_a, vector_x)
    |> Enum.zip(vector_b)
    |> Enum.chunk_every(chunk_size)
    |> Parallel.pmap(fn chunk -> substract_all_rows(chunk) end)
    |> Enum.flat_map(fn x -> x end)
  end

  def square_chunk_x([[]|_]), do: []

  def square_chunk_x([x|[]]) do
    [x*x]
  end

  def square_chunk_x([x|tail]) do
    [x*x]++square_chunk_x(tail)
  end

  def vector_norm(vec, chunk_size) do
    vec
    |> Enum.chunk_every(chunk_size)
    |> Parallel.pmap(fn x->square_chunk_x(x) end)
    |> Enum.flat_map(fn x -> x end)
    |> Enum.sum()
    |> :math.sqrt()
  end
end

defmodule SteepestDescent do
  def solve(matrix_a, b, tolerance) do
    Benchmark.timed_call(fn -> init_loop(matrix_a, b, tolerance) end, "Sequential")
  end

  def init_loop(matrix_a, b, tolerance) do
    x = b
    r = b

    residual = MatrixOps.vector_norm(r)
    if residual < tolerance do
      x
    else
      eval_loop(matrix_a, b, x, r, tolerance)
    end
  end

  def eval_x(x, r, alpha) do
    Enum.map(Enum.zip(x, r), fn {x,r} -> x + alpha*r end)
  end

  def eval_loop(matrix_a, b, x, r, tolerance) do

    alpha = MatrixOps.scalar_product(r, r) / MatrixOps.scalar_product(r, MatrixOps.matvect_product(matrix_a, r))
    r = MatrixOps.gradF(matrix_a, x, b)
    x = eval_x(x,r,alpha)
    residual = MatrixOps.vector_norm(r)

    if residual < tolerance do
      alpha = Benchmark.timed_call(fn -> MatrixOps.scalar_product(r, r) / MatrixOps.scalar_product(r, MatrixOps.matvect_product(matrix_a, r)) end, "seq alpha")
      r = Benchmark.timed_call(fn -> MatrixOps.gradF(matrix_a, x, b) end, "seq r")
      x = Benchmark.timed_call(fn -> eval_x(x,r,alpha) end, "seq x")
      residual = Benchmark.timed_call(fn -> MatrixOps.vector_norm(r) end, "seq residual")

      x
    else
      eval_loop(matrix_a, b, x, r, tolerance)
    end
  end
end

defmodule SteepestDescentParallel do
  def solve(matrix_a, b, tolerance, chunk_size) do
    Benchmark.timed_call(fn -> init_loop(matrix_a, b, tolerance, chunk_size) end, "Parallel")
  end

  def init_loop(matrix_a, b, tolerance, chunk_size) do
    x = b
    r = b

    residual = MatrixOpsParallel.vector_norm(r, chunk_size)
    if residual < tolerance do
      x
    else
      eval_loop(matrix_a, b, x, r, tolerance, chunk_size)
    end
  end

  def eval_x(x, r, alpha, chunk_size) do
    alpha_r = MatrixOpsParallel.const_multiple(alpha, r, chunk_size)
    Enum.zip(x, alpha_r)
    |> Enum.chunk_every(chunk_size)
    |> Parallel.pmap(fn chunk -> MatrixOpsParallel.add_all_rows(chunk) end)
    |> Enum.flat_map(fn x -> x end)
  end

  def eval_loop(matrix_a, b, x, r, tolerance, chunk_size) do
    alpha = MatrixOpsParallel.scalar_product(r, r, chunk_size) / MatrixOpsParallel.scalar_product(r, MatrixOps.matvect_product(matrix_a, r), chunk_size)
    r = MatrixOpsParallel.gradF(matrix_a, x, b, chunk_size)
    x = eval_x(x, r, alpha, chunk_size)
    residual = MatrixOpsParallel.vector_norm(r, chunk_size)

    if residual < tolerance do
      alpha = Benchmark.timed_call(fn -> MatrixOpsParallel.scalar_product(r, r, chunk_size) / MatrixOpsParallel.scalar_product(r, MatrixOps.matvect_product(matrix_a, r), chunk_size) end, "parallle alpha")
      r = Benchmark.timed_call(fn -> MatrixOpsParallel.gradF(matrix_a, x, b, chunk_size) end, "parallel r")
      x = Benchmark.timed_call(fn -> eval_x(x, r, alpha, chunk_size) end, "parallel x")
      residual = Benchmark.timed_call(fn -> MatrixOpsParallel.vector_norm(r, chunk_size) end, "parallel residual")
      x
    else
      eval_loop(matrix_a, b, x, r, tolerance, chunk_size)
    end
  end
end

defmodule Benchmark do
  def measure(function) do
    function
    |> :timer.tc
    |> elem(0)
    #|> Kernel./(1_000_000)
  end

  def timed_call(function,name\\"default name") do
    tstart = DateTime.utc_now()
    result = function.()
    tend = DateTime.utc_now()
    IO.puts "Time spent on op #{name}:#{DateTime.diff(tend,tstart,:millisecond)}"
    result
  end
end



%{matrix: matrix, n: _} = MatrixOps.get_matrix("A.txt")

%{vector: vector, n: _} = MatrixOps.get_vector("B.txt")

_x = SteepestDescent.solve(matrix, vector, 0.1)
# x |> IO.inspect

# :erlang.system_flag(:schedulers_online, 12)
IO.puts inspect :erlang.system_info(:schedulers_online)

chunk_size = 50
_x = SteepestDescentParallel.solve(matrix, vector, 0.1, chunk_size)
