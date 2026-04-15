use std::fmt;
use crate::backend::Backend;
use crate::tensor::Tensor;

/// Activation functions supported by MLRust.
///
/// Each variant is a zero-cost enum tag — the actual computation
/// dispatches through the Backend and is recorded in the compute graph
/// so that backward() can differentiate through it automatically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    /// Identity — passes the pre-activation value straight through.
    /// Useful for the output layer of regression networks.
    Linear,
}

impl Activation {
    /// Apply the activation to a tensor, returning a new tracked tensor.
    pub fn apply<B: Backend>(&self, x: &Tensor<B>) -> Tensor<B> {
        match self {
            Activation::ReLU    => x.relu(),
            Activation::Sigmoid => x.sigmoid(),
            Activation::Tanh    => x.tanh(),
            Activation::Linear  => x.linear_act(),
        }
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Activation::ReLU    => write!(f, "ReLU"),
            Activation::Sigmoid => write!(f, "Sigmoid"),
            Activation::Tanh    => write!(f, "Tanh"),
            Activation::Linear  => write!(f, "Linear"),
        }
    }
}
