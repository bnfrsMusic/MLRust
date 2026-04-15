use crate::backend::Backend;
use crate::tensor::Tensor;

/// A fully-connected (affine) layer:  y = W @ x + b
///
/// Weights initialised with He-normal scaling: N(0, sqrt(2/fan_in)).
pub struct Linear<B: Backend> {
    /// Weight matrix -> shape [out_features, in_features]
    pub weights: Tensor<B>,
    /// Bias vector  -> shape [out_features]
    pub biases: Tensor<B>,
}

impl<B: Backend> Linear<B> {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // He-normal: scale raw normal samples by sqrt(2 / fan_in)
        let scale = (2.0_f32 / in_features as f32).sqrt();
        let w_data: Vec<f32> = B::randn(&[out_features, in_features])
            .iter()
            .map(|v| v * scale)
            .collect();

        let weights = Tensor::from_vec(w_data, vec![out_features, in_features])
            .requires_grad(true);

        let biases = Tensor::zeros(&[out_features])
            .requires_grad(true);

        Self { weights, biases }
    }

    /// Forward pass.
    ///
    /// `input` must be a 1-D tensor of shape [in_features].
    /// returns a 1-D tensor of shape [out_features].
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        // z = W @ x          (matrix-vector product)
        // output = z + bias  (element-wise)
        let z = self.weights.matvec(input);
        &z + &self.biases
    }

    /// All trainable parameters in this layer. used by the optimiser.
    pub fn parameters(&self) -> Vec<&Tensor<B>> {
        vec![&self.weights, &self.biases]
    }

    pub fn in_features(&self) -> usize {
        self.weights.shape()[1]
    }

    pub fn out_features(&self) -> usize {
        self.weights.shape()[0]
    }
}
