package main

import (
	"math/rand"
	"testing"
)

func BenchmarkRandInt(b *testing.B) {
	for i := 0; i < b.N; i++ {
		rand.Int()
	}
}

func BenchmarkMain(bTest *testing.B) {
	tolerance := 0.1

	A, err1 := readMatrixFromFile("./A.txt")
	if err1 != nil {
		panic(err1)
	}
	b, err2 := readVectorFromFile("./B.txt")
	if err2 != nil {
		panic(err2)
	}
	_ = evalSteepestDescentParallel(A, b, tolerance, 10)
}
