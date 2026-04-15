use ndarray::{Array2, ArrayView2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use super::Backend;

/// The CPU backend
/// Uses ndarray and Rust iterators
#[derive(Clone, Debug)]
pub struct Cpu;

impl Backend for Cpu {
    fn randn(shape: &[usize]) -> Vec<f32> {
        let total: usize = shape.iter().product();
        Array2::<f32>::random((total, 1), StandardNormal)
            .into_raw_vec_and_offset()
            .0
    }

    fn matmul(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
        let a_view = ArrayView2::from_shape((m, k), a)
            .expect("matmul: invalid lhs shape");
        let b_view = ArrayView2::from_shape((k, n), b)
            .expect("matmul: invalid rhs shape");
        a_view.dot(&b_view).into_raw_vec_and_offset().0
    }

    fn matvec(mat: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; rows];
        for i in 0..rows {
            let mut s = 0.0f32;
            for j in 0..cols {
                s += mat[i * cols + j] * x[j];
            }
            out[i] = s;
        }
        out
    }

    /// Computes W^T at v   (equivalent to multiplying by the transposed weight matrix)
    fn matvec_t(mat: &[f32], rows: usize, cols: usize, v: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; cols];
        for j in 0..cols {
            let mut s = 0.0f32;
            for i in 0..rows {
                s += mat[i * cols + j] * v[i];
            }
            out[j] = s;
        }
        out
    }

    fn outer(a: &[f32], b: &[f32]) -> Vec<f32> {
        let m = a.len();
        let n = b.len();
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = a[i] * b[j];
            }
        }
        out
    }

    fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = a[i * cols + j];
            }
        }
        out
    }

    fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }

    fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x - y).collect()
    }

    fn mul_elem(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x * y).collect()
    }

    fn scale(a: &[f32], s: f32) -> Vec<f32> {
        a.iter().map(|x| x * s).collect()
    }

    fn neg(a: &[f32]) -> Vec<f32> {
        a.iter().map(|x| -x).collect()
    }

    fn relu(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| v.max(0.0)).collect()
    }

    fn sigmoid(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
    }

    fn tanh_act(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| v.tanh()).collect()
    }

    fn sum_all(x: &[f32]) -> f32 {
        x.iter().sum()
    }
}
