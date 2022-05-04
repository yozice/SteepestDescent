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
	maxGoRoutines := 12

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
	x = evalSteepestDescent(A, b, tolerance)
	duration := time.Since(start)
	fmt.Println("Sequential", duration)

	start = time.Now()
	x = evalSteepestDescentParallel(A, b, tolerance, maxGoRoutines)
	duration = time.Since(start)
	fmt.Println("Concurrent", duration)

	writeRoots(x, "result.txt")
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
	copy(x, b)
	copy(r, b)

	residual = math.Sqrt(evalScalarProductParallel(r, r, maxGoRoutines))
	for tolerance < residual {
		alpha = evalScalarProductParallel(r, r, maxGoRoutines) / evalScalarProductParallel(r, evalMatrixVectorProductParallel(A, r, maxGoRoutines), maxGoRoutines)

		r = gradFParallel(A, x, b, maxGoRoutines)
		x = evalXParallel(x, alpha, r, maxGoRoutines)

		residual = math.Sqrt(evalScalarProductParallel(r, r, maxGoRoutines))
	}

	return x
}

type empty struct{}

func evalXParallel(x []float64, alpha float64, r []float64, maxGoRoutines int) []float64 {
	var wg sync.WaitGroup
	n := len(x)

	var blockSize int
	switch maxGoRoutines {
	case 1:
		blockSize = n
	case 2:
		blockSize = n/2 + 1
	default:
		blockSize = n / maxGoRoutines
	}

	for i := 0; i < n; i += blockSize {
		if i+2*blockSize > n {
			blockSize = n - i
		}
		wg.Add(1)

		go func(i int, blockSize int) {
			defer wg.Done()

			for k := i; k < i+blockSize; k++ {
				x[k] = x[k] + alpha*r[k]
			}
		}(i, blockSize)
	}
	wg.Wait()
	return x
}

func evalScalarProduct(x []float64, y []float64) float64 {
	product := 0.0

	for i := 0; i < len(x); i++ {
		product += x[i] * y[i]
	}

	return product

}

func evalScalarProductParallel(x []float64, y []float64, maxGoRoutines int) float64 {
	product := 0.0
	n := len(x)
	var wg sync.WaitGroup
	var blockSize int
	switch maxGoRoutines {
	case 1:
		blockSize = n
	case 2:
		blockSize = n/2 + 1
	default:
		blockSize = n / maxGoRoutines
	}
	productCh := make(chan float64, maxGoRoutines)
	for i := 0; i < n; i += blockSize {
		if i+2*blockSize > n {
			blockSize = n - i
		}
		wg.Add(1)

		go func(i int, blockSize int, productCh chan float64) {
			defer wg.Done()

			buf := 0.0

			for k := i; k < i+blockSize; k++ {
				buf += x[k] * y[k]
			}
			productCh <- buf

		}(i, blockSize, productCh)
	}
	wg.Wait()

	close(productCh)

	for val := range productCh {
		product += val
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

func evalMatrixVectorProductParallel(A [][]float64, x []float64, maxGoRoutines int) []float64 {
	var wg sync.WaitGroup
	n := len(A)
	var blockSize int
	switch maxGoRoutines {
	case 1:
		blockSize = n
	case 2:
		blockSize = n/2 + 1
	default:
		blockSize = n / maxGoRoutines
	}

	result := make([]float64, len(A))
	for i := 0; i < n; i += blockSize {
		if i+2*blockSize > n {
			blockSize = n - i
		}
		wg.Add(1)

		go func(i int, blockSize int) {
			defer wg.Done()

			for k := i; k < i+blockSize; k++ {
				for j := 0; j < n; j++ {
					result[k] += A[k][j] * x[j]
				}
			}

		}(i, blockSize)
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

func gradFParallel(A [][]float64, x []float64, b []float64, maxGoRoutines int) []float64 {
	n := len(x)
	y := evalMatrixVectorProduct(A, x)
	var wg sync.WaitGroup
	var blockSize int
	switch maxGoRoutines {
	case 1:
		blockSize = n
	case 2:
		blockSize = n/2 + 1
	default:
		blockSize = n / maxGoRoutines
	}

	for i := 0; i < n; i += blockSize {
		if i+2*blockSize > n {
			blockSize = n - i
		}
		wg.Add(1)

		go func(i int, blockSize int) {
			defer wg.Done()

			for k := i; k < i+blockSize; k++ {
				y[k] = b[k] - y[k]
			}
		}(i, blockSize)
	}
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
