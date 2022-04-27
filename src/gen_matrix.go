package main

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {
	n := 100
	A := generateMatrix(10, -10, n, n)
	B := generateMatrix(10, -10, 1, n)
	A = generateMatrixPD(A)
	writeFile(A, n, "./A.txt")
	writeFile(B, n, "./B.txt")

}

func generateMatrix(max int, min int, n int, m int) [][]float64 {
	rand.Seed(time.Now().UnixNano())

	var s [][]float64

	for i := 0; i < n; i++ {
		s = append(s, make([]float64, 0))
		for j := 0; j < m; j++ {
			s[i] = append(s[i], getRandom(max, min))
		}
	}

	return s
}

func generateMatrixPD(matrix [][]float64) [][]float64 {
	n := len(matrix)
	transposed := matrix
	pdMatrix := make([][]float64, n)
	for i := range pdMatrix {
		pdMatrix[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			transposed[i][j] = matrix[j][i]
		}
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			for k := 0; k < n; k++ {
				pdMatrix[i][j] += transposed[i][k] * matrix[k][j]
			}
		}
	}
	return pdMatrix
}

func writeFile(matrix [][]float64, n int, filename string) {
	file, err := os.Create(filename)

	if err != nil {
		fmt.Println("Unable to create file:", err)
		os.Exit(1)
	}
	defer file.Close()
	file.WriteString(strconv.Itoa(n) + "\n")
	for _, u := range matrix {
		file.WriteString(strings.Trim(strings.Join(strings.Fields(fmt.Sprint(u)), " "), "[]") + "\n")
	}
}

func getRandom(max int, min int) float64 {
	return float64(rand.Intn(max-min)+min) + rand.Float64()
}
