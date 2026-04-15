use std::fmt;
use crate::backend::Backend;
use crate::tensor::Tensor;

///Loss functions
///Currently supported loss functions: MSE, BCE
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loss {
    /// Mean Squared Error
    Mse,
    /// Binary Cross-Entropy
    Bce,
}

impl Loss {
    /// Compute loss between `predictions` and `targets`.
    ///
    /// Both must have the same shape.  Returns a scalar tensor.
    pub fn compute<B: Backend>(
        &self,
        predictions: &Tensor<B>,
        targets: &Tensor<B>,
    ) -> Tensor<B> {
        match self {
            Loss::Mse => predictions.mse_loss(targets),
            Loss::Bce => predictions.bce_loss(targets),
        }
    }
}


/// 
impl fmt::Display for Loss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Loss::Mse => write!(f, "MSE"),
            Loss::Bce => write!(f, "BCE"),
        }
    }
}
