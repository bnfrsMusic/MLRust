use std::cell::RefCell;
use crate::backend::Backend;

// ---------------------------------------------------------------------------
// Thread-local graph -> one graph per thread, zero synchronisation overhead
// during a forward pass. perfect for single-threaded training loops.
// ---------------------------------------------------------------------------
thread_local! {
    pub(crate) static GRAPH: RefCell<ComputeGraph> = RefCell::new(ComputeGraph::new());
}

// ---------------------------------------------------------------------------
//index into the arena
// ---------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(pub(crate) usize);

// ---------------------------------------------------------------------------
// the actual data that lives in the arena.
// ---------------------------------------------------------------------------
pub(crate) struct TensorStorage {
    pub(crate) data: Vec<f32>,
    pub(crate) shape: Vec<usize>,
    pub(crate) grad: Option<Vec<f32>>,
    pub(crate) requires_grad: bool,
}

// ---------------------------------------------------------------------------
// Op -> records which operation produced a tensor and from which inputs.
// This is all the information needed to compute gradients in backward().
// ---------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub(crate) enum Op {
    /// Leaf tensors: user-created weights, inputs, targets.
    Leaf,

    // ---- Elementwise -------------------------------------------------------
    Add(TensorId, TensorId),
    Sub(TensorId, TensorId),
    ElemMul(TensorId, TensorId),
    Neg(TensorId),
    Scale(TensorId, f32),

    // ---- Linear algebra ----------------------------------------------------
    /// 2D @ 2D.  lhs:[m,k]  rhs:[k,n]  out:[m,n]
    MatMul(TensorId, TensorId),
    /// W[rows,cols] @ x[cols] = y[rows]
    MatVec(TensorId, TensorId),

    // ---- Activations -------------------------------------------------------
    Relu(TensorId),
    Sigmoid(TensorId),
    Tanh(TensorId),

    // ---- Reductions --------------------------------------------------------
    /// Sum all elements to a scalar.
    Sum(TensorId),

    // ---- Loss functions (return scalar) ------------------------------------
    MseLoss(TensorId, TensorId),
    BceLoss(TensorId, TensorId),
}

// ---------------------------------------------------------------------------
// ComputeGraph: arena that owns all tensor storage and the op graph.
// ---------------------------------------------------------------------------
pub(crate) struct ComputeGraph {
    pub(crate) storage: Vec<TensorStorage>,
    pub(crate) ops: Vec<Op>,
}

impl ComputeGraph {
    pub(crate) fn new() -> Self {
        Self {
            storage: Vec::new(),
            ops: Vec::new(),
        }
    }

    /// Allocate new tensor in arena and return its stable ID.
    pub(crate) fn alloc(
        &mut self,
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
        op: Op,
    ) -> TensorId {
        let id = TensorId(self.storage.len());
        self.storage.push(TensorStorage {
            data,
            shape,
            grad: None,
            requires_grad,
        });
        self.ops.push(op);
        id
    }

    ///clear only the gradients of the specified tensors 
    pub(crate) fn zero_grad_for(&mut self, ids: &[TensorId]) {
        for &id in ids {
            self.storage[id.0].grad = None;
        }
    }

    /// discard all transient (non-parameter) tensors from the arena and
    /// compact parameter tensors into a fresh arena at low indices.
    /// MUST be called between training iterations.
    /// returns a mapping old_id -> new_id so callers can patch their handles
    pub(crate) fn reset_transient(&mut self, keep: &[TensorId]) -> Vec<(TensorId, TensorId)> {
        let mut new_storage: Vec<TensorStorage> = Vec::with_capacity(keep.len());
        let mut new_ops: Vec<Op> = Vec::with_capacity(keep.len());
        let mut id_map: Vec<(TensorId, TensorId)> = Vec::with_capacity(keep.len());

        for &old_id in keep {
            let new_id = TensorId(new_storage.len());
            id_map.push((old_id, new_id));

            let s = &self.storage[old_id.0];
            new_storage.push(TensorStorage {
                data: s.data.clone(),
                shape: s.shape.clone(),
                grad: None,          //gradients always reset on compaction
                requires_grad: s.requires_grad,
            });
            new_ops.push(Op::Leaf); //parameters are leaves in every new sub-graph
        }

        self.storage = new_storage;
        self.ops = new_ops;
        id_map
    }

    // -----------------------------------------------------------------------
    // Accumulate a gradient into a tensor's grad slot.
    // -----------------------------------------------------------------------
    fn accumulate_grad(&mut self, id: TensorId, grad: Vec<f32>) {
        let s = &mut self.storage[id.0];
        if !s.requires_grad {
            return;
        }
        match &mut s.grad {
            Some(existing) => {
                for (e, g) in existing.iter_mut().zip(grad.iter()) {
                    *e += g;
                }
            }
            None => {
                s.grad = Some(grad);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Backward pass
    // -----------------------------------------------------------------------
    pub(crate) fn backward<B: Backend>(&mut self, root: TensorId) {
        // Seed the root gradient.
        let root_len = self.storage[root.0].data.len();
        self.storage[root.0].grad = Some(vec![1.0f32; root_len]);

        for i in (0..=root.0).rev() {
            // Clone gradient and op so we can mutably borrow storage later.
            let grad = match self.storage[i].grad.clone() {
                Some(g) => g,
                None => continue,
            };
            let op = self.ops[i].clone();

            match op {
                Op::Leaf => { /* gradient has arrived at a leaf -> nothing to propagate */ }

                // ---- Elementwise -------------------------------------------
                Op::Add(lhs, rhs) => {
                    self.accumulate_grad(lhs, grad.clone());
                    self.accumulate_grad(rhs, grad);
                }

                Op::Sub(lhs, rhs) => {
                    self.accumulate_grad(lhs, grad.clone());
                    let neg: Vec<f32> = grad.iter().map(|&g| -g).collect();
                    self.accumulate_grad(rhs, neg);
                }

                Op::ElemMul(lhs, rhs) => {
                    let ld = self.storage[lhs.0].data.clone();
                    let rd = self.storage[rhs.0].data.clone();
                    let gl: Vec<f32> = grad.iter().zip(rd.iter()).map(|(g, r)| g * r).collect();
                    let gr: Vec<f32> = grad.iter().zip(ld.iter()).map(|(g, l)| g * l).collect();
                    self.accumulate_grad(lhs, gl);
                    self.accumulate_grad(rhs, gr);
                }

                Op::Neg(input) => {
                    let ng: Vec<f32> = grad.iter().map(|&g| -g).collect();
                    self.accumulate_grad(input, ng);
                }

                Op::Scale(input, scalar) => {
                    let sg: Vec<f32> = grad.iter().map(|&g| g * scalar).collect();
                    self.accumulate_grad(input, sg);
                }

                // ---- Linear algebra ----------------------------------------
                // C = A @ B   A:[m,k]  B:[k,n]  C:[m,n]
                // ∂L/∂A = ∂L/∂C @ B^T    [m,n] @ [n,k] = [m,k]
                // ∂L/∂B = A^T @ ∂L/∂C    [k,m] @ [m,n] = [k,n]
                Op::MatMul(lhs, rhs) => {
                    let ls = self.storage[lhs.0].shape.clone(); // [m,k]
                    let rs = self.storage[rhs.0].shape.clone(); // [k,n]
                    let ld = self.storage[lhs.0].data.clone();
                    let rd = self.storage[rhs.0].data.clone();
                    let (m, k, n) = (ls[0], ls[1], rs[1]);

                    let b_t = B::transpose(&rd, k, n);             // [n,k]
                    let grad_a = B::matmul(&grad, m, n, &b_t, k); // [m,k]
                    let a_t = B::transpose(&ld, m, k);             // [k,m]
                    let grad_b = B::matmul(&a_t, k, m, &grad, n); // [k,n]

                    self.accumulate_grad(lhs, grad_a);
                    self.accumulate_grad(rhs, grad_b);
                }

                // y = W @ x   W:[out,in]  x:[in]  y:[out]
                // ∂L/∂W = ∂L/∂y ⊗ x          outer product [out,in]
                // ∂L/∂x = W^T @ ∂L/∂y         [in,out] @ [out] = [in]
                Op::MatVec(mat, vec) => {
                    let ms = self.storage[mat.0].shape.clone(); // [out,in]
                    let md = self.storage[mat.0].data.clone();
                    let xd = self.storage[vec.0].data.clone();
                    let (out, inp) = (ms[0], ms[1]);

                    let grad_w = B::outer(&grad, &xd);               // [out,in]
                    let grad_x = B::matvec_t(&md, out, inp, &grad);  // [in]

                    self.accumulate_grad(mat, grad_w);
                    self.accumulate_grad(vec, grad_x);
                }

                // ---- Activations -------------------------------------------
                // ReLU: f'(x) = 1 if output > 0 else 0
                Op::Relu(input) => {
                    let od = self.storage[i].data.clone(); // post-activation output
                    let gi: Vec<f32> = od.iter().zip(grad.iter())
                        .map(|(&o, &g)| if o > 0.0 { g } else { 0.0 })
                        .collect();
                    self.accumulate_grad(input, gi);
                }

                // Sigmoid: f'(x) = f(x) * (1 - f(x))
                Op::Sigmoid(input) => {
                    let od = self.storage[i].data.clone();
                    let gi: Vec<f32> = od.iter().zip(grad.iter())
                        .map(|(&o, &g)| g * o * (1.0 - o))
                        .collect();
                    self.accumulate_grad(input, gi);
                }

                // Tanh: f'(x) = 1 - f(x)^2
                Op::Tanh(input) => {
                    let od = self.storage[i].data.clone();
                    let gi: Vec<f32> = od.iter().zip(grad.iter())
                        .map(|(&o, &g)| g * (1.0 - o * o))
                        .collect();
                    self.accumulate_grad(input, gi);
                }

                // ---- reductions --------------------------------------------
                // Sum: gradient fans out equally to every input element.
                Op::Sum(input) => {
                    let n = self.storage[input.0].data.len();
                    let gi = vec![grad[0]; n];
                    self.accumulate_grad(input, gi);
                }

                // ---- loss functions ----------------------------------------
                // MSE = mean((p - t)^2)   ∂L/∂p = 2(p - t) / n
                Op::MseLoss(pred, target) => {
                    let pd = self.storage[pred.0].data.clone();
                    let td = self.storage[target.0].data.clone();
                    let n = pd.len() as f32;
                    let gp: Vec<f32> = pd.iter().zip(td.iter())
                        .map(|(&p, &t)| grad[0] * 2.0 * (p - t) / n)
                        .collect();
                    self.accumulate_grad(pred, gp);
                }

                // BCE = -mean(t*log(p) + (1-t)*log(1-p))
                // ∂L/∂p = (-t/p + (1-t)/(1-p)) / n
                Op::BceLoss(pred, target) => {
                    const EPS: f32 = 1e-7;
                    let pd = self.storage[pred.0].data.clone();
                    let td = self.storage[target.0].data.clone();
                    let n = pd.len() as f32;
                    let gp: Vec<f32> = pd.iter().zip(td.iter())
                        .map(|(&p, &t)| {
                            let pc = p.max(EPS).min(1.0 - EPS);
                            grad[0] * (-t / pc + (1.0 - t) / (1.0 - pc)) / n
                        })
                        .collect();
                    self.accumulate_grad(pred, gp);
                }
            }
        }
    }
}