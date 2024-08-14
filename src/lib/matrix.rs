use rand::{thread_rng, Rng};

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
} 


impl Matrix {
    pub fn zeros(rows:usize, cols:usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],//2D array of rows x cols filled with 0.0         
        }
    
    }

    pub fn random(rows: usize, cols: usize) -> Matrix{
        let mut rng = thread_rng();
        let mut res = Matrix::zeros(rows,cols);

        for i in 0..rows {
            for j in 0..cols {
            
                res.data[i][j] = rng.gen::<f64>() * 2.0 - 1.0;//Gives rand value between -1 and 1

            }
        }
        
        res // returns res after nested loop
    }

    pub fn from (data: Vec<Vec<f64>>) -> Matrix {//converts 2D vector to Matrix
        Matrix {

            rows: data.len(),
            cols: data[0].len(),
            data

        }
    }

    pub fn multiply(&self, other: &Matrix) -> Matrix {
        if self.cols != other.rows { // Checks if matrix multiplication possible
            panic!("Attempted to multiply by matrix of incompatible dimensions");
        }

        //result calculation (Matrix Multiplication)
        let mut res = Matrix::zeros(self.rows, other.cols);

        for i in 0..self.rows {

            for j in 0..other.cols {

                let mut sum = 0.0;

                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }

                res.data[i][j] = sum;
            }
        }
        res//returns result
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        if (self.rows != other.rows) || (self.cols != other.cols) { // Checks if matrix addition possible
            //println!("{} x {} and {} x {}", self.rows, self.cols, other.rows, other.cols);
            panic!("Attempted to add matrices of incompatible dimensions");
        }

        let mut res = Matrix::zeros(self.rows, self.cols);
        
        for i in 0..self.rows{
            for j in 0..self.cols{
                //Addition
                res.data[i][j] = self.data[i][j] + other.data[i][j];
            }

            
        } 
        res//returns result
    }

    pub fn dot_multiply(&mut self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols { // Checks if matrix dot multiplication possible
            println!("{} x {} and {} x {}", self.rows, self.cols, other.rows, other.cols);
            panic!("Attempted to dot multiply matrices of incompatible dimensions");
        }

        let mut res = Matrix::zeros(self.rows, self.cols);
        
        for i in 0..self.rows{
            for j in 0..self.cols{
                //Multiplication
                res.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }
            res//returns result
    }

    pub fn substract(&self, other: &Matrix) -> Matrix {
        if self.rows != other.rows || self.cols != other.cols { // Checks if matrix subtraction possible
            println!("{} x {} and {} x {}", self.rows, self.cols, other.rows, other.cols);
            panic!("Attempted to substract matrices of incompatible dimensions");
        }

        let mut res = Matrix::zeros(self.rows, self.cols);
        
        for i in 0..self.rows{
            for j in 0..self.cols{
                //Addition
                res.data[i][j] = self.data[i][j] - other.data[i][j];
            }

            
        } 
        res//returns result
    }

    pub fn map(&mut self, function: &dyn Fn(f64) -> f64) -> Matrix { // takes every individual element and then applies a function
    
        Matrix::from(
            (self.data)
            .clone()
            .into_iter()
            .map(|row| row.into_iter().map(|value| function(value)).collect())
            .collect(),
        )
    }

    pub fn transpose(&mut self) -> Matrix {
        //turn cols into rows and rows into cols
        let mut res = Matrix::zeros(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                
                res.data[j][i] = self.data[i][j];

            }
        }
        res
    }


}