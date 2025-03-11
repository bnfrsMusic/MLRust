use std::ops::{Add, Mul, Sub};
use rand::Rng;

#[derive(Clone, Debug)]
pub struct Tensor {
    data: Vec<Vec<f64>>,
}

impl Tensor {
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        // Validate that all rows have the same length
        if !data.is_empty() {
            let first_row_len = data[0].len();
            for row in &data {
                assert_eq!(row.len(), first_row_len, "All rows must have the same length");
            }
        }
        
        Tensor { data }
    }
    
    // Create a tensor filled with zeros
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let data = vec![vec![0.0; cols]; rows];
        Tensor { data }
    }
    
    // Create a tensor filled with random values
    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(rows);
        
        for _ in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for _ in 0..cols {
                // Initialize with small random values between -0.5 and 0.5
                row.push(rng.gen::<f64>() - 0.5);
            }
            data.push(row);
        }
        
        Tensor { data }
    }
    
    // Get the shape of the tensor
    pub fn shape(&self) -> (usize, usize) {
        if self.data.is_empty() {
            return (0, 0);
        }
        (self.data.len(), self.data[0].len())
    }
    
    // Get a specific value from the tensor
    pub fn get_value(&self, row: usize, col: usize) -> f64 {
        self.data[row][col]
    }
    
    // Set a specific value in the tensor
    pub fn set_value(&mut self, row: usize, col: usize, value: f64) {
        self.data[row][col] = value;
    }
    
    // Get a row from the tensor
    pub fn get_row(&self, row: usize) -> Vec<f64> {
        self.data[row].clone()
    }
    
    // Transpose the tensor
    pub fn transpose(&self) -> Tensor {
        let (rows, cols) = self.shape();
        let mut result = Tensor::zeros(cols, rows);
        
        for i in 0..rows {
            for j in 0..cols {
                result.set_value(j, i, self.get_value(i, j));
            }
        }
        
        result
    }
    
    // Matrix multiplication
    pub fn dot(&self, other: &Tensor) -> Tensor {
        let (self_rows, self_cols) = self.shape();
        let (other_rows, other_cols) = other.shape();
        
        assert_eq!(self_cols, other_rows, "Incompatible dimensions for dot product");
        
        let mut result = Tensor::zeros(self_rows, other_cols);
        
        for i in 0..self_rows {
            for j in 0..other_cols {
                let mut sum = 0.0;
                for k in 0..self_cols {
                    sum += self.get_value(i, k) * other.get_value(k, j);
                }
                result.set_value(i, j, sum);
            }
        }
        
        result
    }
    
    // Element-wise multiplication (Hadamard product)
    pub fn hadamard(&self, other: &Tensor) -> Tensor {
        let (self_rows, self_cols) = self.shape();
        let (other_rows, other_cols) = other.shape();
        
        assert_eq!(self_rows, other_rows, "Incompatible dimensions for Hadamard product");
        assert_eq!(self_cols, other_cols, "Incompatible dimensions for Hadamard product");
        
        let mut result = Tensor::zeros(self_rows, self_cols);
        
        for i in 0..self_rows {
            for j in 0..self_cols {
                let value = self.get_value(i, j) * other.get_value(i, j);
                result.set_value(i, j, value);
            }
        }
        
        result
    }
    
    // Apply a function to each element of the tensor
    pub fn map<F>(&self, f: F) -> Tensor 
    where F: Fn(f64) -> f64 {
        let (rows, cols) = self.shape();
        let mut result = Tensor::zeros(rows, cols);
        
        for i in 0..rows {
            for j in 0..cols {
                let value = f(self.get_value(i, j));
                result.set_value(i, j, value);
            }
        }
        
        result
    }
    
    // Sum along columns
    pub fn sum_axis_0(&self) -> Tensor {
        let (rows, cols) = self.shape();
        let mut result = Tensor::zeros(1, cols);
        
        for j in 0..cols {
            let mut sum = 0.0;
            for i in 0..rows {
                sum += self.get_value(i, j);
            }
            result.set_value(0, j, sum);
        }
        
        result
    }
    
    // Sum along rows
    pub fn sum_axis_1(&self) -> Tensor {
        let (rows, cols) = self.shape();
        let mut result = Tensor::zeros(rows, 1);
        
        for i in 0..rows {
            let mut sum = 0.0;
            for j in 0..cols {
                sum += self.get_value(i, j);
            }
            result.set_value(i, 0, sum);
        }
        
        result
    }
}

// Implement addition for Tensor
impl Add for &Tensor {
    type Output = Tensor;
    
    fn add(self, other: &Tensor) -> Tensor {
        let (self_rows, self_cols) = self.shape();
        let (other_rows, other_cols) = other.shape();
        
        assert_eq!(self_rows, other_rows, "Incompatible dimensions for addition");
        assert_eq!(self_cols, other_cols, "Incompatible dimensions for addition");
        
        let mut result = Tensor::zeros(self_rows, self_cols);
        
        for i in 0..self_rows {
            for j in 0..self_cols {
                let value = self.get_value(i, j) + other.get_value(i, j);
                result.set_value(i, j, value);
            }
        }
        
        result
    }
}

// Implement subtraction for Tensor
impl Sub for &Tensor {
    type Output = Tensor;
    
    fn sub(self, other: &Tensor) -> Tensor {
        let (self_rows, self_cols) = self.shape();
        let (other_rows, other_cols) = other.shape();
        
        assert_eq!(self_rows, other_rows, "Incompatible dimensions for subtraction");
        assert_eq!(self_cols, other_cols, "Incompatible dimensions for subtraction");
        
        let mut result = Tensor::zeros(self_rows, self_cols);
        
        for i in 0..self_rows {
            for j in 0..self_cols {
                let value = self.get_value(i, j) - other.get_value(i, j);
                result.set_value(i, j, value);
            }
        }
        
        result
    }
}

// Implement scalar multiplication
impl Mul<f64> for &Tensor {
    type Output = Tensor;
    
    fn mul(self, scalar: f64) -> Tensor {
        let (rows, cols) = self.shape();
        let mut result = Tensor::zeros(rows, cols);
        
        for i in 0..rows {
            for j in 0..cols {
                let value = self.get_value(i, j) * scalar;
                result.set_value(i, j, value);
            }
        }
        
        result
    }
}