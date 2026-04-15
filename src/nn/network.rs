use std::collections::HashMap;
use crate::backend::Backend;
use crate::tensor::Tensor;
use crate::nn::{Linear, Activation};
use crate::graph::TensorId;

struct LayerEntry<B: Backend> {
    linear: Linear<B>,
    activation: Activation,
}

///Sequential feed-forward network.
pub struct Network<B: Backend> {
    pub(crate) layers: Vec<LayerEntry<B>>,
}

impl<B: Backend> Network<B> {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer(
        mut self,
        in_features: usize,
        out_features: usize,
        activation: Activation,
    ) -> Self {
        self.layers.push(LayerEntry {
            linear: Linear::new(in_features, out_features),
            activation,
        });
        self
    }

    /// Forward pass. Input must be a 1-D tensor matching the first layer's in_features.
    pub fn forward(&self, input: &Tensor<B>) -> Tensor<B> {
        let mut x = input.clone();
        for entry in &self.layers {
            let z = entry.linear.forward(&x);
            x = entry.activation.apply(&z);
        }
        x
    }

    /// All trainable parameters across every layer, in layer order.
    pub fn parameters(&self) -> Vec<&Tensor<B>> {
        self.layers
            .iter()
            .flat_map(|e| e.linear.parameters())
            .collect()
    }

    ///collect TensorIds of all parameters in a stable order (weights then biases per layer)
    pub(crate) fn parameter_ids(&self) -> Vec<TensorId> {
        self.layers
            .iter()
            .flat_map(|e| vec![e.linear.weights.id, e.linear.biases.id])
            .collect()
    }

    ///After a graph reset, patch every stored TensorId to its new compact position.
    pub(crate) fn remap_ids(&mut self, mapping: &[(TensorId, TensorId)]) {
        let map: HashMap<usize, usize> = mapping.iter()
            .map(|&(old, new)| (old.0, new.0))
            .collect();

        for entry in &mut self.layers {
            let w = &mut entry.linear.weights;
            if let Some(&ni) = map.get(&w.id.0) {
                w.id = TensorId(ni);
            }
            let b = &mut entry.linear.biases;
            if let Some(&ni) = map.get(&b.id.0) {
                b.id = TensorId(ni);
            }
        }
    }

    ///compact the arena down to only this network's parameters, patch the
    /// network's own TensorIds, and zero the gradients
    /// MUST be called at END of each training iteration (after opt.step())
    pub fn end_iter(&mut self) {
        let ids = self.parameter_ids();
        let mapping = crate::graph::GRAPH.with(|g| {
            { let mut gr = g.borrow_mut(); gr.reset_transient(&ids) }
        });
        self.remap_ids(&mapping);
    }

    /// Print a human-readable summary of the network architecture.
    pub fn print_summary(&self) {
        println!("╔══════════════════════════════════════╗");
        println!("║         MLRust Network Summary       ║");
        println!("╠══════════════════════════════════════╣");
        for (i, entry) in self.layers.iter().enumerate() {
            let lin = &entry.linear;
            println!(
                "║ Layer {:2}  [{:4} → {:4}]  {:8}  ║",
                i + 1,
                lin.in_features(),
                lin.out_features(),
                entry.activation.to_string(),
            );
        }
        println!("╠══════════════════════════════════════╣");
        println!("║ Total parameters: {:18} ║", self.count_params());
        println!("╚══════════════════════════════════════╝");
    }

    fn count_params(&self) -> usize {
        self.layers.iter().map(|e| {
            let ws = e.linear.weights.shape();
            let bs = e.linear.biases.shape();
            ws.iter().product::<usize>() + bs.iter().product::<usize>()
        }).sum()
    }
}

impl<B: Backend> Default for Network<B> {
    fn default() -> Self {
        Self::new()
    }
}
