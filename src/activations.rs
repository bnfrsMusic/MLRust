use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Sigmoid,
    ReLU,
    TanH,
    Linear,
}

impl Activation {
    // Apply the activation function to a tensor
    pub fn apply(&self, x: &Tensor) -> Tensor {
        match self {
            Activation::Sigmoid => x.map(|val| 1.0 / (1.0 + (-val).exp())),
            Activation::ReLU => x.map(|val| if val > 0.0 { val } else { 0.0 }),
            Activation::TanH => x.map(|val| val.tanh()),
            Activation::Linear => x.clone(),
        }
    }
    
    // Calculate the derivative of the activation function
    // This is applied to the already activated values (output of the activation function)
    pub fn derivative(&self, activated_values: &Tensor) -> Tensor {
        match self {
            Activation::Sigmoid => {
                // For sigmoid: f'(x) = f(x) * (1 - f(x))
                activated_values.map(|val| val * (1.0 - val))
            },
            Activation::ReLU => {
                // For ReLU: f'(x) = 1 if x > 0, 0 otherwise
                activated_values.map(|val| if val > 0.0 { 1.0 } else { 0.0 })
            },
            Activation::TanH => {
                // For tanh: f'(x) = 1 - f(x)^2
                activated_values.map(|val| 1.0 - val * val)
            },
            Activation::Linear => {
                // For linear: f'(x) = 1
                activated_values.map(|_| 1.0)
            },
        }
    }
}