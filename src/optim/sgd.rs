use crate::backend::Backend;
use crate::tensor::Tensor;
use crate::graph::{GRAPH, TensorId};

/// Stochastic Gradient Descent optimiser.
///
/// How to use:
/// ```rust,ignore
/// let opt = Sgd::new(0.01);
/// // inside training loop:
/// opt.zero_grad(&net.parameters());
/// loss.backward();
/// opt.step(&net.parameters());
/// ```
pub struct Sgd {
    pub learning_rate: f32,
}

impl Sgd {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }

    /// Update parameters: θ <- θ - η * ∂L/∂θ
    ///Call this after `loss.backward()`.
    pub fn step<B: Backend>(&self, parameters: &[&Tensor<B>]) {
        GRAPH.with(|g| {
            let mut gr = g.borrow_mut();
            let lr = self.learning_rate;
            for param in parameters {
                // Clone grad first to avoid double-borrow
                if let Some(grad) = gr.storage[param.id.0].grad.clone() {
                    let data = &mut gr.storage[param.id.0].data;
                    for (w, g) in data.iter_mut().zip(grad.iter()) {
                        *w -= lr * g;
                    }
                }
            }
        });
    }

    /// 0 the gradients of given parameters.
    /// Call this before each forward pass to prevent gradient accumulation
    /// across iterations.
    pub fn zero_grad<B: Backend>(&self, parameters: &[&Tensor<B>]) {
        GRAPH.with(|g| {
            let mut gr = g.borrow_mut();
            for param in parameters {
                gr.storage[param.id.0].grad = None;
            }
        });
    }

    /// correct full training step:
    ///   1. compact the arena (drops all activation tensors from the last iteration)
    ///   2. re-map the network's internal TensorIds to their new positions
    ///   3. zeros gradients on the surviving parameter tensors
    ///
    /// Call this at the END of each iteration (after step()), not the beginning
    /// Returns the new id mapping so Network can update itself
    pub fn reset_graph(&self, param_ids: &[TensorId]) -> Vec<(TensorId, TensorId)> {
        GRAPH.with(|g| {
            { let mut gr = g.borrow_mut(); gr.reset_transient(param_ids) }
        })
    }
}