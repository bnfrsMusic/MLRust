use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Sub, Mul, Neg};
use crate::backend::Backend;
use crate::graph::{GRAPH, TensorId, Op};

// ---------------------------------------------------------------------------
// Tensor<B>: the only type users ever touch
// just a u32-sized index into the thread-local arena.
// ---------------------------------------------------------------------------
pub struct Tensor<B: Backend> {
    pub(crate) id: TensorId,
    _backend: PhantomData<B>,
}

// Cloning a Tensor only copies the lightweight handle
impl<B: Backend> Clone for Tensor<B> {
    fn clone(&self) -> Self {
        Self { id: self.id, _backend: PhantomData }
    }
}

impl<B: Backend> Tensor<B> {
    // -----------------------------------------------------------------------
    // (internal helper) build a Tensor from a raw TensorId
    // -----------------------------------------------------------------------
    pub(crate) fn from_id(id: TensorId) -> Self {
        Self { id, _backend: PhantomData }
    }

    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// 0
    pub fn zeros(shape: &[usize]) -> Self {
        let data = vec![0.0f32; shape.iter().product()];
        GRAPH.with(|g| {
            let id = g.borrow_mut().alloc(data, shape.to_vec(), false, Op::Leaf);
            Self::from_id(id)
        })
    }

    /// 1
    pub fn ones(shape: &[usize]) -> Self {
        let data = vec![1.0f32; shape.iter().product()];
        GRAPH.with(|g| {
            let id = g.borrow_mut().alloc(data, shape.to_vec(), false, Op::Leaf);
            Self::from_id(id)
        })
    }

    /// RNG as given from the backend
    pub fn randn(shape: &[usize]) -> Self {
        let data = B::randn(shape);
        GRAPH.with(|g| {
            let id = g.borrow_mut().alloc(data, shape.to_vec(), false, Op::Leaf);
            Self::from_id(id)
        })
    }

    ///build from an existing Vec. Shape must match the data length.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "from_vec: data length {} doesn't match shape {:?}",
            data.len(), shape
        );
        GRAPH.with(|g| {
            let id = g.borrow_mut().alloc(data, shape, false, Op::Leaf);
            Self::from_id(id)
        })
    }

    // -----------------------------------------------------------------------
    // Grad tracking
    // -----------------------------------------------------------------------

    /// Mark this tensor as requiring a gradient.  Returns self for chaining.
    pub fn requires_grad(self, rg: bool) -> Self {
        GRAPH.with(|g| {
            g.borrow_mut().storage[self.id.0].requires_grad = rg;
        });
        self
    }

    pub fn is_requires_grad(&self) -> bool {
        GRAPH.with(|g| g.borrow().storage[self.id.0].requires_grad)
    }

    // -----------------------------------------------------------------------
    //Data access
    // -----------------------------------------------------------------------

    /// Clone underlying data out of the arena.
    pub fn data(&self) -> Vec<f32> {
        GRAPH.with(|g| g.borrow().storage[self.id.0].data.clone())
    }

    pub fn shape(&self) -> Vec<usize> {
        GRAPH.with(|g| g.borrow().storage[self.id.0].shape.clone())
    }

    /// Retrieve the accumulated gradient after backward().
    pub fn grad(&self) -> Option<Vec<f32>> {
        GRAPH.with(|g| g.borrow().storage[self.id.0].grad.clone())
    }

    /// Extract a scalar value. Panics if the tensor has more than one element.
    pub fn item(&self) -> f32 {
        GRAPH.with(|g| {
            let g = g.borrow();
            let d = &g.storage[self.id.0].data;
            assert_eq!(d.len(), 1, "item() on tensor with {} elements. use data() instead", d.len());
            d[0]
        })
    }

    // -----------------------------------------------------------------------
    //Autograd entry-point
    // -----------------------------------------------------------------------

    /// run backward pass from this tensor
    pub fn backward(&self) {
        GRAPH.with(|g| g.borrow_mut().backward::<B>(self.id));
    }

    // -----------------------------------------------------------------------
    // Forward operations
    // Each op: (1) read inputs from the arena, (2) compute the result,
    // (3) allocate the result in the arena with correct Op tag.
    // -----------------------------------------------------------------------

    /// matrix-vector product: self is [out, in], other is [in] -> [out]
    pub fn matvec(&self, other: &Tensor<B>) -> Tensor<B> {
        GRAPH.with(|g| {
            let mut gr = g.borrow_mut();
            let wd = gr.storage[self.id.0].data.clone();
            let ws = gr.storage[self.id.0].shape.clone();
            let xd = gr.storage[other.id.0].data.clone();

            assert_eq!(ws.len(), 2, "matvec: weight must be 2D, got {:?}", ws);
            assert_eq!(
                xd.len(), ws[1],
                "matvec shape mismatch: [{},{}] @ [{}]", ws[0], ws[1], xd.len()
            );

            let result = B::matvec(&wd, ws[0], ws[1], &xd);
            let rg = gr.storage[self.id.0].requires_grad
                   || gr.storage[other.id.0].requires_grad;
            let id = gr.alloc(result, vec![ws[0]], rg, Op::MatVec(self.id, other.id));
            Tensor::from_id(id)
        })
    }

    /// 2-D matrix multiplication: self is [m,k], other is [k,n] -> [m,n]
    pub fn matmul(&self, other: &Tensor<B>) -> Tensor<B> {
        GRAPH.with(|g| {
            let mut gr = g.borrow_mut();
            let ad = gr.storage[self.id.0].data.clone();
            let ash = gr.storage[self.id.0].shape.clone();
            let bd = gr.storage[other.id.0].data.clone();
            let bsh = gr.storage[other.id.0].shape.clone();

            assert_eq!(ash.len(), 2, "matmul: lhs must be 2D, got {:?}", ash);
            assert_eq!(bsh.len(), 2, "matmul: rhs must be 2D, got {:?}", bsh);
            assert_eq!(
                ash[1], bsh[0],
                "matmul shape mismatch: [{},{}] @ [{},{}]",
                ash[0], ash[1], bsh[0], bsh[1]
            );

            let (m, k, n) = (ash[0], ash[1], bsh[1]);
            let result = B::matmul(&ad, m, k, &bd, n);
            let rg = gr.storage[self.id.0].requires_grad
                   || gr.storage[other.id.0].requires_grad;
            let id = gr.alloc(result, vec![m, n], rg, Op::MatMul(self.id, other.id));
            Tensor::from_id(id)
        })
    }

    // ---- Activation Functions -------------------------------------------------------

    pub fn relu(&self) -> Tensor<B> {
        GRAPH.with(|g| {
            let mut gr = g.borrow_mut();
            let d = gr.storage[self.id.0].data.clone();
            let s = gr.storage[self.id.0].shape.clone();
            let result = B::relu(&d);
            let rg = gr.storage[self.id.0].requires_grad;
            let id = gr.alloc(result, s, rg, Op::Relu(self.id));
            Tensor::from_id(id)
        })
    }

    pub fn sigmoid(&self) -> Tensor<B> {
        GRAPH.with(|g| {
            let mut gr = g.borrow_mut();
            let d = gr.storage[self.id.0].data.clone();
            let s = gr.storage[self.id.0].shape.clone();
            let result = B::sigmoid(&d);
            let rg = gr.storage[self.id.0].requires_grad;
            let id = gr.alloc(result, s, rg, Op::Sigmoid(self.id));
            Tensor::from_id(id)
        })
    }

    pub fn tanh(&self) -> Tensor<B> {
        GRAPH.with(|g| {
            let mut gr = g.borrow_mut();
            let d = gr.storage[self.id.0].data.clone();
            let s = gr.storage[self.id.0].shape.clone();
            let result = B::tanh_act(&d);
            let rg = gr.storage[self.id.0].requires_grad;
            let id = gr.alloc(result, s, rg, Op::Tanh(self.id));
            Tensor::from_id(id)
        })
    }

    /// returns handle to the same tensor. 
    pub fn linear_act(&self) -> Tensor<B> {
        self.clone()
    }

    // ---- Reductions --------------------------------------------------------

    ///sum all elements to a scalar tensor.
    pub fn sum(&self) -> Tensor<B> {
        GRAPH.with(|g| {
            let mut gr = g.borrow_mut();
            let d = gr.storage[self.id.0].data.clone();
            let s: f32 = B::sum_all(&d);
            let rg = gr.storage[self.id.0].requires_grad;
            let id = gr.alloc(vec![s], vec![1], rg, Op::Sum(self.id));
            Tensor::from_id(id)
        })
    }

    // ---- Loss functions ----------------------------------------------------

    /// Mean Squared Error loss: mean((self - target)^2)
    pub fn mse_loss(&self, target: &Tensor<B>) -> Tensor<B> {
        GRAPH.with(|g| {
            let mut gr = g.borrow_mut();
            let pd = gr.storage[self.id.0].data.clone();
            let td = gr.storage[target.id.0].data.clone();
            assert_eq!(pd.len(), td.len(), "mse_loss: shape mismatch");

            let n = pd.len() as f32;
            let loss: f32 = pd.iter().zip(td.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f32>() / n;

            let rg = gr.storage[self.id.0].requires_grad;
            let id = gr.alloc(vec![loss], vec![1], rg, Op::MseLoss(self.id, target.id));
            Tensor::from_id(id)
        })
    }

    /// Binary Cross-Entropy loss: -mean(t*log(p) + (1-t)*log(1-p))
    pub fn bce_loss(&self, target: &Tensor<B>) -> Tensor<B> {
        GRAPH.with(|g| {
            let mut gr = g.borrow_mut();
            let pd = gr.storage[self.id.0].data.clone();
            let td = gr.storage[target.id.0].data.clone();
            assert_eq!(pd.len(), td.len(), "bce_loss: shape mismatch");

            const EPS: f32 = 1e-7;
            let n = pd.len() as f32;
            let loss: f32 = pd.iter().zip(td.iter())
                .map(|(p, t)| {
                    let pc = p.max(EPS).min(1.0 - EPS);
                    -(t * pc.ln() + (1.0 - t) * (1.0 - pc).ln())
                })
                .sum::<f32>() / n;

            let rg = gr.storage[self.id.0].requires_grad;
            let id = gr.alloc(vec![loss], vec![1], rg, Op::BceLoss(self.id, target.id));
            Tensor::from_id(id)
        })
    }
}

// ---------------------------------------------------------------------------
//Operator overloading
// ---------------------------------------------------------------------------

impl<B: Backend> Add<&Tensor<B>> for &Tensor<B> {
    type Output = Tensor<B>;
    fn add(self, rhs: &Tensor<B>) -> Tensor<B> {
        GRAPH.with(|g| {
            let mut gr = g.borrow_mut();
            let a = gr.storage[self.id.0].data.clone();
            let b = gr.storage[rhs.id.0].data.clone();
            assert_eq!(a.len(), b.len(),
                "Add shape mismatch: {:?} vs {:?}",
                gr.storage[self.id.0].shape, gr.storage[rhs.id.0].shape);
            let result = B::add(&a, &b);
            let shape = gr.storage[self.id.0].shape.clone();
            let rg = gr.storage[self.id.0].requires_grad
                   || gr.storage[rhs.id.0].requires_grad;
            let id = gr.alloc(result, shape, rg, Op::Add(self.id, rhs.id));
            Tensor::from_id(id)
        })
    }
}

impl<B: Backend> Sub<&Tensor<B>> for &Tensor<B> {
    type Output = Tensor<B>;
    fn sub(self, rhs: &Tensor<B>) -> Tensor<B> {
        GRAPH.with(|g| {
            let mut gr = g.borrow_mut();
            let a = gr.storage[self.id.0].data.clone();
            let b = gr.storage[rhs.id.0].data.clone();
            let result = B::sub(&a, &b);
            let shape = gr.storage[self.id.0].shape.clone();
            let rg = gr.storage[self.id.0].requires_grad
                   || gr.storage[rhs.id.0].requires_grad;
            let id = gr.alloc(result, shape, rg, Op::Sub(self.id, rhs.id));
            Tensor::from_id(id)
        })
    }
}

impl<B: Backend> Mul<&Tensor<B>> for &Tensor<B> {
    type Output = Tensor<B>;
    fn mul(self, rhs: &Tensor<B>) -> Tensor<B> {
        GRAPH.with(|g| {
            let mut gr = g.borrow_mut();
            let a = gr.storage[self.id.0].data.clone();
            let b = gr.storage[rhs.id.0].data.clone();
            let result = B::mul_elem(&a, &b);
            let shape = gr.storage[self.id.0].shape.clone();
            let rg = gr.storage[self.id.0].requires_grad
                   || gr.storage[rhs.id.0].requires_grad;
            let id = gr.alloc(result, shape, rg, Op::ElemMul(self.id, rhs.id));
            Tensor::from_id(id)
        })
    }
}

impl<B: Backend> Neg for &Tensor<B> {
    type Output = Tensor<B>;
    fn neg(self) -> Tensor<B> {
        GRAPH.with(|g| {
            let mut gr = g.borrow_mut();
            let d = gr.storage[self.id.0].data.clone();
            let s = gr.storage[self.id.0].shape.clone();
            let result = B::neg(&d);
            let rg = gr.storage[self.id.0].requires_grad;
            let id = gr.alloc(result, s, rg, Op::Neg(self.id));
            Tensor::from_id(id)
        })
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------
impl<B: Backend> fmt::Display for Tensor<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        GRAPH.with(|g| {
            let gr = g.borrow();
            let s = &gr.storage[self.id.0];
            let preview: Vec<String> = s.data.iter()
                .take(8)
                .map(|v| format!("{:.4}", v))
                .collect();
            let ellipsis = if s.data.len() > 8 { ", ..." } else { "" };
            write!(f, "Tensor(shape={:?}, data=[{}{}])", s.shape, preview.join(", "), ellipsis)
        })
    }
}

impl<B: Backend> fmt::Debug for Tensor<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
