defmodule Parallel do
  def pmap(collection, func) do
    collection
    |> Enum.map(&(Task.async(fn -> func.(&1) end)))
    |> Enum.map(fn x -> Task.await(x,100000) end)
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

  #dot product for two vectors
  def scalar_product(a,b) do
    Enum.zip(a,b) |> Parallel.pmap(fn {x,y} -> x*y end) |> Enum.sum()
  end

  def matvect_product(matrix_a, vector_x) do
    Parallel.pmap(matrix_a, fn(row) -> scalar_product(row, vector_x) end)
  end

  def gradF(matrix_a, vector_x, vector_b) do
    matvect_product(matrix_a, vector_x) |> Enum.zip(vector_b) |> Parallel.pmap(fn {y,b} -> b-y end)
  end

  def vector_norm(vec) do
    Parallel.pmap(vec,fn x->x*x end) |> Enum.sum() |> :math.sqrt()
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

  def eval_loop(matrix_a, b, x, r, tolerance) do
    alpha = MatrixOps.scalar_product(r, r) / MatrixOps.scalar_product(r, MatrixOps.matvect_product(matrix_a, r))
    r = MatrixOps.gradF(matrix_a, x, b)
    # IO.puts(r)
    x = Enum.map(Enum.zip(x, r), fn {x,r} -> x + alpha*r end)
    residual = MatrixOps.vector_norm(r)

    if residual < tolerance do
      x
    else
      eval_loop(matrix_a, b, x, r, tolerance)
    end
  end
end

defmodule SteepestDescentParallel do
  def solve(matrix_a, b, tolerance) do
    Benchmark.timed_call(fn -> init_loop(matrix_a, b, tolerance) end, "Sequential")
  end

  def init_loop(matrix_a, b, tolerance) do
    x = b
    r = b

    residual = MatrixOpsParallel.vector_norm(r)
    if residual < tolerance do
      x
    else
      eval_loop(matrix_a, b, x, r, tolerance)
    end
  end

  def eval_loop(matrix_a, b, x, r, tolerance) do
    alpha = MatrixOpsParallel.scalar_product(r, r) / MatrixOpsParallel.scalar_product(r, MatrixOpsParallel.matvect_product(matrix_a, r))
    r = MatrixOpsParallel.gradF(matrix_a, x, b)
    # IO.puts(r)
    x = Parallel.pmap(Enum.zip(x, r), fn {x,r} -> x + alpha*r end)
    residual = MatrixOpsParallel.vector_norm(r)

    if residual < tolerance do
      x
    else
      eval_loop(matrix_a, b, x, r, tolerance)
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

_x = SteepestDescentParallel.solve(matrix, vector, 0.1)
# x |> IO.inspect
# x = MatrixOps.matvect_product(matrix, vector)
# x |> IO.puts

# IO.inspect matrix, label: "the list is"
# IO.inspect vector, label: "the list is"


# IO.inspect matrix, label: "the list is"
# [head | tail] = matrix
# IO.puts length(matrix)
# IO.puts length(head)
