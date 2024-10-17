use std::thread::panicking;

use rand::{thread_rng, Rng};
//use std::default::Default;

#[derive(Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Tensor {
    //A row-priority Tensor that runs on CPU
    //[0.0,1.0,2.0,3.0] with shape [2,2] is the same thing as:
    //[0.0, 1.0]
    //[2.0, 3.0]
    pub fn new(shape: Vec<usize>) -> Tensor {
        let mut data = Vec::new();
        let mut count = 1;
        let mut rng = thread_rng();

        for i in &shape {
            count *= i;
        }
        for _ in 0..count {
            data.push(0.0);
        }

        Tensor { shape, data }
    }
    pub fn get(&self, index: Vec<usize>) -> f64 {
        assert!(
            !self.shape.is_empty(),
            "Shape is empty, cannot perform get operation"
        );
        assert!(
            index.len() == self.shape.len(),
            "Index length {:?} does not match shape length {:?}",
            index.len(),
            self.shape.len()
        );

        let mut res = 0;
        let mut stride = 1;

        for i in (0..index.len()).rev() {
            assert!(
                index[i] < self.shape[i],
                "Index out of bounds for dimension {}",
                i
            );
            res += index[i] * stride;
            stride *= self.shape[i];
        }

        assert!(
            res < self.data.len(),
            "Calculated index {} is out of bounds for data with length {}",
            res,
            self.data.len()
        );

        self.data[res]
    }

    pub fn set(&mut self, index: Vec<usize>, value: f64) {
        assert!(
            !self.shape.is_empty(),
            "Shape is empty, cannot perform set operation"
        );
        assert!(
            index.len() == self.shape.len(),
            "Index length {:?} does not match shape length {:?}",
            index.len(),
            self.shape.len()
        );

        let mut res = 0;
        let mut stride = 1;

        for i in (0..index.len()).rev() {
            assert!(
                index[i] < self.shape[i],
                "Index out of bounds for dimension {}",
                i
            );
            res += index[i] * stride;
            stride *= self.shape[i];
        }

        assert!(
            res < self.data.len(),
            "Calculated index {} is out of bounds for data with length {}",
            res,
            self.data.len()
        );

        self.data[res] = value;
    }

    pub fn return_vector(&self) -> Vec<f64> {
        self.data.clone()
    }

    pub fn random(shape: Vec<usize>) -> Tensor {
        let mut rng = thread_rng();
        let mut res = Tensor::new(shape);

        rng.gen::<f64>();

        for i in 0..res.data.len() {
            res.data[i] = rng.gen::<f64>() * 2.0 - 1.0;
        }

        res
    }

    pub fn from(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
        //converts vector to tensor
        Tensor { shape, data }
    }

    pub fn map(&mut self, function: &dyn Fn(f64) -> f64) -> Tensor {
        let new_data: Vec<f64> = self.data.iter().map(|&value| function(value)).collect();
        Tensor::from(self.shape.clone(), new_data)
    }

    pub fn transpose(&self) -> Tensor {
        assert!(self.shape.len() == 2, "Transpose only supports 2D tensors");

        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut transposed_data = vec![0.0; self.data.len()];

        for i in 0..rows {
            for j in 0..cols {
                transposed_data[j * rows + i] = self.get(vec![i, j]);
            }
        }

        Tensor::from(vec![cols, rows], transposed_data)
    }

    //--------------------------------------------------------------Multiplication---------------------------------------------------------------------

    pub fn multiply(&self, other: &Tensor) -> Tensor {
        let self_shape_len = self.shape.len();
        let other_shape_len = other.shape.len();
        //println!("{:?} and {:?}", self_shape_len, other_shape_len);
        /*
        println!(
            "self shape: {:?} \n other shape: {:?}",
            self.shape, other.shape
        );*/
        if (other_shape_len == 1 && self_shape_len > other_shape_len) {
            let mut res = Tensor::new(vec![self.shape[0], other.shape[1]]);
            panic!(
                "TYPE 1, self shape: {:?} \n other shape: {:?}",
                self.shape, other.shape
            );
            return res;
        }

        if (self_shape_len == 1 && self_shape_len < other_shape_len) {
            panic!(
                "TYPE 2, self shape: {:?} \n other shape: {:?}",
                self.shape, other.shape
            );
        }

        // Check if Strassen's algorithm is applicable

        if self_shape_len == 2 && other_shape_len == 2 && self.shape == other.shape {
            let res = self.strassen_multiply(other);
            //println!("shape: {:?} data: {:?}", res.shape, res.data);
            return res;
        }

        // Check if both tensors are 2D
        if self_shape_len == 2
            && other_shape_len == 2
            && (self.shape[1] == other.shape[0]
                || self.shape[0] == self.shape[1]
                || self.shape[0] == self.shape[0])
        {
            let res = self.matrix_multiply(other);
            //println!("shape: {:?} data: {:?}", res.shape, res.data);
            return res;
        }
        //Panic if incompatible dimensionsions.
        panic!(
            "NO MULTIPLICATION POSSIBLE. INCOMPATIBLE DIMENSIONS {:?} AND {:?}",
            self.shape, other.shape
        )
    }

    fn matrix_multiply(&self, other: &Tensor) -> Tensor {
        assert!(self.shape.len() == 2 && other.shape.len() == 2);
        assert!(
            self.shape[1] == other.shape[0] || self.shape == other.shape, // || self.shape[0] == other.shape[0]
            "Incompatible shapes ({:?} and {:?})for matrix multiplication",
            self.shape,
            other.shape
        );

        let mut res = Tensor::new(vec![self.shape[0], other.shape[1]]);

        for i in 0..self.shape[0] {
            for j in 0..other.shape[1] {
                let mut sum = 0.0;
                for k in 0..self.shape[1] {
                    sum += self.get(vec![i, k]) * other.get(vec![k, j]);
                }
                res.set(vec![i, j], sum);
            }
        }

        res
    }

    fn strassen_multiply(&self, other: &Tensor) -> Tensor {
        // Simplified version of Strassen's Algorithm for 2x2 matrices
        if self.shape == vec![2, 2] && other.shape == vec![2, 2] {
            let a = self.get(vec![0, 0]);
            let b = self.get(vec![0, 1]);
            let c = self.get(vec![1, 0]);
            let d = self.get(vec![1, 1]);

            let e = other.get(vec![0, 0]);
            let f = other.get(vec![0, 1]);
            let g = other.get(vec![1, 0]);
            let h = other.get(vec![1, 1]);

            let p1 = a * (f - h);
            let p2 = (a + b) * h;
            let p3 = (c + d) * e;
            let p4 = d * (g - e);
            let p5 = (a + d) * (e + h);
            let p6 = (b - d) * (g + h);
            let p7 = (a - c) * (e + f);

            let mut res = Tensor::new(vec![2, 2]);
            res.set(vec![0, 0], p5 + p4 - p2 + p6);
            res.set(vec![0, 1], p1 + p2);
            res.set(vec![1, 0], p3 + p4);
            res.set(vec![1, 1], p1 + p5 - p3 - p7);

            return res;
        } else {
            // If not 2x2, use regular multiplication
            return self.matrix_multiply(other); //
        }
    }

    pub fn dot(&self, other: &Tensor) -> Tensor {
        //Dot multiplication and inner product
        if other.shape != self.shape {
            panic!("Attempt to dot multiply tensors of incompatible dimensions");
        }

        let mut ten = Vec::with_capacity(self.shape.len());
        for &dim in &self.shape {
            ten.push(dim);
        }

        let mut res = Tensor::new(ten);

        for i in 0..self.data.len() {
            res.data[i] = self.data[i] * self.data[i]
        }

        res
    }

    pub fn multiply_scalar(&mut self, scalar: f64) -> Tensor {
        self.map(&|x| x * scalar)
    }

    //--------------------------------------------------------------Addition and Substraction---------------------------------------------------------------------

    pub fn add(&mut self, other: &Tensor) {
        assert!(
            self.shape == other.shape || other.shape.len() == 1 && other.shape[0] == self.shape[0],
            "Shape mismatch for addition: {:?} and {:?}",
            self.shape,
            other.shape
        );

        if other.shape.len() == 1 && self.shape.len() == 2 {
            // Broadcasting the biases across the second dimension
            for i in 0..self.shape[0] {
                for j in 0..self.shape[1] {
                    self.data[i * self.shape[1] + j] += other.data[i];
                }
            }
        } else {
            for i in 0..self.data.len() {
                self.data[i] += other.data[i];
            }
        }
    }

    pub fn substract(&mut self, other: &Tensor) {
        assert!(
            self.shape == other.shape || other.shape.len() == 1 && other.shape[0] == self.shape[0],
            "Shape mismatch for subtraction: {:?} and {:?} with data {:?} and {:?}",
            self.shape,
            other.shape,
            self.data,
            other.data
        );

        if other.shape.len() == 1 && self.shape.len() == 2 {
            // Broadcasting the biases across the second dimension
            for i in 0..self.shape[0] {
                for j in 0..self.shape[1] {
                    self.data[i * self.shape[1] + j] -= other.data[i];
                }
            }
        } else {
            for i in 0..self.data.len() {
                self.data[i] -= other.data[i];
            }
        }
    }

    pub fn increase_dim(&mut self, amt: usize) {
        for _i in 0..amt {
            self.shape.push(1);
        }
    }
}
