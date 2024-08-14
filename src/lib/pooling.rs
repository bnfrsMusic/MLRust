use super::tensor::Tensor;

#[derive(Clone, Debug)]
pub enum PoolingType {
    Max,
    Average,
}

pub struct Pooling {
    pooling_type: PoolingType,
    window_size: Tensor,
    stride: usize,
}

impl Pooling {
    pub fn new(pooling_type: PoolingType, window_size: Tensor, stride: usize) -> Self {
        Pooling {
            pooling_type,
            window_size,
            stride,
        }
    }

    pub fn apply(self, input: Tensor) -> Tensor {
        let mut output_shape = Vec::new();
        let num_dims = &input.shape.len();
        let window_shape = &self.window_size.shape;

        for (i, &dim) in input.clone().shape.iter().enumerate() {
            //calculates the remaining dimensions after pooling
            let output_dim = (dim - window_shape[i] + self.stride) / self.stride;
            output_shape.push(output_dim);
        }
        let mut res = Tensor::new(output_shape);
        let window_size = window_shape[0];

        res
    }
}
