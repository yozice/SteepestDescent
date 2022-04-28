package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

func main() {
	var x []float64
	tolerance := 0.1
	maxGoRoutines := 5

	A, err1 := readMatrixFromFile("./A.txt")
	if err1 != nil {
		panic(err1)
	}
	b, err2 := readVectorFromFile("./B.txt")
	if err2 != nil {
		panic(err2)
	}
	err := checkSymmetry(A)
	if err != true {
		print("Not symmetrical")
		return
	}
	start := time.Now()
	// x := evalSteepestDescent(A, b, tolerance)
	duration := time.Since(start)
	fmt.Println("Sequential", duration)

	start = time.Now()
	x = evalSteepestDescentParallel(A, b, tolerance, maxGoRoutines)
	duration = time.Since(start)
	fmt.Println("Concurrent", duration)

	writeRoots(x, "result.txt")
	// fmt.Println(A, b)

	// x := [][]float64{{1,2,3},{1,2,3}}
	// fmt.Println(evalScalarProduct(x,x))
}

func evalSteepestDescent(A [][]float64, b []float64, tolerance float64) []float64 {
	n := len(b)
	var alpha float64
	x := make([]float64, n)
	r := make([]float64, n)
	var residual float64

	copy(x, b)
	copy(r, b)

	residual = math.Sqrt(evalScalarProduct(r, r))
	for tolerance < residual {
		alpha = evalScalarProduct(r, r) / evalScalarProduct(r, evalMatrixVectorProduct(A, r))
		r = gradF(A, x, b)
		for i := 0; i < n; i++ {
			x[i] = x[i] + alpha*r[i]
		}

		residual = math.Sqrt(evalScalarProduct(r, r))
	}

	return x
}

func evalSteepestDescentParallel(A [][]float64, b []float64, tolerance float64, maxGoRoutines int) []float64 {
	n := len(b)
	var alpha float64
	x := make([]float64, n)
	r := make([]float64, n)
	var residual float64
	sem := make(chan int)
	var wg sync.WaitGroup

	copy(x, b)
	copy(r, b)

	residual = math.Sqrt(evalScalarProductParallel(r, r))
	for tolerance < residual {

		go func() {
			alpha = evalScalarProductParallel(r, r) / evalScalarProductParallel(r, evalMatrixVectorProductParallel(A, r))
			sem <- 0
		}()

		go func() {
			r = gradF(A, x, b)
			sem <- 0
		}()
		<-sem
		<-sem

		wg.Add(n)
		for i := 0; i < n; i++ {
			go func(i int) {
				x[i] = x[i] + alpha*r[i]
				wg.Done()
			}(i)

		}
		residual = math.Sqrt(evalScalarProductParallel(r, r))
		wg.Wait()
	}

	return x
}

func evalScalarProduct(x []float64, y []float64) float64 {
	product := 0.0

	for i := 0; i < len(x); i++ {
		product += x[i] * y[i]
	}

	return product

}

func evalScalarProductParallel(x []float64, y []float64) float64 {
	sem := make(chan float64)
	product := 0.0
	n := len(x)

	for i := 0; i < n; i++ {
		go func(i int) {
			sem <- x[i] * y[i]
		}(i)
	}
	for i := 0; i < n; i++ {
		product += <-sem
	}
	return product

}

func evalMatrixVectorProduct(A [][]float64, x []float64) []float64 {
	n := len(A)

	result := make([]float64, len(A))

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			result[i] += A[i][j] * x[j]
		}
	}

	return result
}

func evalMatrixVectorProductParallel(A [][]float64, x []float64) []float64 {
	n := len(A)
	var wg sync.WaitGroup

	result := make([]float64, len(A))
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			for j := 0; j < n; j++ {
				result[i] += A[i][j] * x[j]
			}
			wg.Done()
		}(i)
	}
	wg.Wait()
	return result
}

func gradF(A [][]float64, x []float64, b []float64) []float64 {
	y := evalMatrixVectorProduct(A, x)
	for i := 0; i < len(x); i++ {
		y[i] = b[i] - y[i]
	}

	return y
}

func gradFParallel(A [][]float64, x []float64, b []float64) []float64 {
	// sem := make(chan int)
	var wg sync.WaitGroup

	n := len(x)
	y := evalMatrixVectorProduct(A, x)

	wg.Add(n)
	for i := 0; i < n; i++ {
		go func(i int) {
			y[i] = b[i] - y[i]
			wg.Done()
		}(i)
	}
	// for i := 0; i < n; i++ {
	// 	<-sem
	// }
	wg.Wait()

	return y
}

func writeRoots(vector []float64, filename string) {
	file, err := os.Create(filename)
	n := len(vector)
	if err != nil {
		fmt.Println("Unable to create file:", err)
		os.Exit(1)
	}
	defer file.Close()
	file.WriteString(strconv.Itoa(n) + "\n")
	file.WriteString(strings.Trim(strings.Join(strings.Fields(fmt.Sprint(vector)), " "), "[]") + "\n")
}

func checkSymmetry(A [][]float64) bool {
	n := len(A)
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			if A[i][j] != A[j][i] {
				return false
			}
		}
	}
	return true
}

func readVectorFromFile(path string) ([]float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Scan()
	m, err := strconv.Atoi(scanner.Text())

	vector := make([]float64, m)

	i := 0
	for scanner.Scan() {
		str_arr := strings.Split(scanner.Text(), " ")
		for j, val := range str_arr {
			vector[j], err = strconv.ParseFloat(val, 64)
		}
		i += 1
	}

	return vector, scanner.Err()
}

func readMatrixFromFile(path string) ([][]float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Scan()
	m, err := strconv.Atoi(scanner.Text())

	matrix := make([][]float64, m)
	for i := range matrix {
		matrix[i] = make([]float64, m)
	}

	i := 0
	for scanner.Scan() {
		str_arr := strings.Split(scanner.Text(), " ")
		for j, val := range str_arr {
			matrix[i][j], err = strconv.ParseFloat(val, 64)
		}
		i += 1
	}

	return matrix, scanner.Err()
}
