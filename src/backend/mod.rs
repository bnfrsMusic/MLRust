pub mod cpu;
pub use cpu::Cpu;

/// The Backend trait abstracts all compute operations.
pub trait Backend: Clone + Send + Sync + 'static {
    ///random normal samples 
    fn randn(shape: &[usize]) -> Vec<f32>;

    /// 2D matmul: A[m,k] @ B[k,n] -> C[m,n]
    fn matmul(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32>;

    ///Matrix-vector product: W[rows,cols] @ x[cols] -> y[rows]
    fn matvec(mat: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32>;

    ///Transposed matrix-vector product: W^T[cols,rows] @ v[rows] -> y[cols]
    /// Used in backward pass for gradient computation
    fn matvec_t(mat: &[f32], rows: usize, cols: usize, v: &[f32]) -> Vec<f32>;

    ///Outer product: a[m] ⊗ b[n] -> C[m,n]
    /// Used in backward pass for weight gradient computation
    fn outer(a: &[f32], b: &[f32]) -> Vec<f32>;

    ///transpose: A[rows,cols] -> A^T[cols,rows]
    fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32>;

    fn add(a: &[f32], b: &[f32]) -> Vec<f32>;
    fn sub(a: &[f32], b: &[f32]) -> Vec<f32>;
    fn mul_elem(a: &[f32], b: &[f32]) -> Vec<f32>;
    fn scale(a: &[f32], s: f32) -> Vec<f32>;
    fn neg(a: &[f32]) -> Vec<f32>;
    fn relu(x: &[f32]) -> Vec<f32>;
    fn sigmoid(x: &[f32]) -> Vec<f32>;
    fn tanh_act(x: &[f32]) -> Vec<f32>;
    fn sum_all(x: &[f32]) -> f32;
}
