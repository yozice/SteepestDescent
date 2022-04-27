package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

func main() {
	nCPU := runtime.NumCPU()
	runtime.GOMAXPROCS(nCPU)

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
	x := evalSteepestDescent(A, b, tolerance)
	duration := time.Since(start)
	fmt.Println("Linear", duration)

	start = time.Now()
	x = evalSteepestDescentParallel(A, b, tolerance, maxGoRoutines)
	duration = time.Since(start)
	fmt.Println("Parallel", duration)

	writeRoots(x, "result.txt")
	// fmt.Println(A, b)

	// x := [][]float64{{1,2,3},{1,2,3}}
	// fmt.Println(evalScalarProduct(x,x))
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

func evalSteepestDescent(A [][]float64, b []float64, tolerance float64) []float64 {
	n := len(b)
	var alpha float64
	x := make([]float64, n)
	r := make([]float64, n)
	var residual float64

	copy(x, b)
	copy(r, b)
	// x = b
	// r = b
	residual = math.Sqrt(evalScalarProduct(r, r))
	for tolerance < residual {
		// rr := evalScalarProduct(r, r)
		// ar := evalMatrixVectorProduct(A, r)
		// rar := evalScalarProduct(r, ar)
		// alpha = rr / rar
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
	// var wg sync.WaitGroup

	copy(x, b)
	copy(r, b)
	// x = b
	// r = b
	residual = math.Sqrt(evalScalarProductParallel(r, r))
	for tolerance < residual {
		// wg.Add(2)
		// go func() {
		// 	defer wg.Done()
		alpha = evalScalarProductParallel(r, r) / evalScalarProductParallel(r, evalMatrixVectorProductParallel(A, r))
		// }()

		// go func() {
		// defer wg.Done()
		r = gradF(A, x, b)
		// }()
		// wg.Wait()
		for i := 0; i < n; i++ {
			// wg.Add(1)
			// go func(i int) {
			// 	defer wg.Done()
			// 	x[i] = x[i] + alpha*r[i]
			// }(i)
			x[i] = x[i] + alpha*r[i]

		}
		// wg.Wait()
		residual = math.Sqrt(evalScalarProductParallel(r, r))

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
	product := 0.0
	var wg sync.WaitGroup

	for i := 0; i < len(x); i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			product += x[i] * y[i]
		}(i)
	}
	wg.Wait()

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

	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < n; j++ {
				result[i] += A[i][j] * x[j]
			}
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
	var wg sync.WaitGroup

	y := evalMatrixVectorProduct(A, x)
	for i := 0; i < len(x); i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			y[i] = b[i] - y[i]
		}(i)
	}
	wg.Wait()

	return y
}
