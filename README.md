# GPU-Ready LDPC Neural Decoder Training

This repository contains a PyTorch-based implementation of an LDPC neural decoder designed to run efficiently on GPUs. It includes:

* **LDPC Generator/Parity matrix construction in GF(2)**
* **GPU-accelerated dataset creation** with BPSK modulation and AWGN noise
* **Multi-branch neural network architecture** incorporating syndrome-based features
* **Training loop and BER computation** fully on GPU

---

## ✨ Key Features

* Handles full message space for `k=14` (16,384 codewords)
* Real-time generation of noisy codewords on GPU
* Multi-input neural decoder using:

  * Raw channel output `Y`
  * Sign-based quantization `signY`
  * Parity-syndrome representation `Hr_sign`
  * Projected syndrome `HrY`
* Easily replaceable LDPC construction (supports custom H / G matrices)
* BER evaluation under multiple noise variances

---

## 📦 Requirements

```bash
python>=3.9
torch>=2.0
numpy
matplotlib
tqdm
```

Ensure CUDA is available for full acceleration.

---

## 🚀 Running Training

```bash
python gpu_ready_ldpc_training.py
```

Training begins with 10 epochs and prints epoch-wise loss. Adjust epoch count in the script:

```python
EPOCHS = 10
```

---

## 🎯 BER Evaluation

After training, the script automatically measures BER over different noise variances:

```python
variances_test = [0.1,0.2,0.3,0.4,0.5]
```

Output similar to:

```
Variance 0.1: BER=1.23e-4
Variance 0.2: BER=2.84e-4
...
```

A plot of **BER vs Noise variance** will also be generated.

---

## 🧠 Model Architecture

The multi-branch network structure:

```
(Y) ------> Dense -> Dense -> Dense -----\
(HrY) ---> Dense -> Dense ---------------+-> Fusion -> Output bits
(signY) -> Dense -> Dense ---------------+
(Hr_sign)-> Dense -> Dense --------------/
```

Output logits are fed into `BCEWithLogitsLoss`.

---

## 📁 File Structure

```
├── gpu_ready_ldpc_training.py   # Main training + BER script
└── README.md                     # Documentation file
```

---

## 📌 Notes / Tips

* Set `desired_k` to control number of message bits
* Increase `realizations_per_var` for more robust BER evaluation
* Custom LDPC codes can be inserted by replacing `gm_generation()`

---

## 🧾 License

MIT License

---

## 🤝 Contributing

Pull requests and improvements are welcome!

---

## 📮 Contact

For queries or discussions related to LDPC, ML-based decoding, or wireless research:
**Mahesh Nagulapalli**

---

### ⭐ If you find this useful, consider starring the repository!
