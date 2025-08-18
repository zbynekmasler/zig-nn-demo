# Zig Neural Network Demo (Public Preview)

> **Status:** Public demo
> The **full repository** (modules, history, experiments, renderers) is available **on request**.

## ✨ What this is

A minimal, **from‑scratch** feed‑forward neural network in **Zig** that shows:

* **Fundamentals** — manual forward pass + backprop (SGD)
* **Performance** — cache‑friendly loops, no hidden allocations
* **Portability** — Linux/macOS/Windows; edge‑device friendly

Built on the same NN core I use for **probabilistic motion prediction** and **telco‑style user state classification**.

---

## 🧭 Why Zig

* **Low‑level control:** deterministic memory, predictable performance
* **Transparency:** every float, loop, and branch is visible
* **Production‑lean:** small binaries, no heavyweight ML deps

For Dataclair, this demonstrates **first‑principles AI engineering**, hot‑loop optimization, and clean integration points to Python/CSV/IPC when needed.

---

## 🧱 Core features

* Fully manual NN: weights, biases, forward, backprop (SGD), ReLU hidden layer
* Binary weight **save/load** (`weights.bin`)
* Clean modular code, ready to extend (CSV adapters, Vulkan visualizer, batching)

---

## 🧠 Network topology

```mermaid
graph TD;
    subgraph "Input Layer"
        I1["Input 1 x_A"]
        I2["Input 2 y_A"]
        I3["Input 3 x_B"]
        I4["Input 4 y_B"]
    end

    subgraph "Hidden Layer ReLU"
        H1["Neuron H1 Σ w*x + b → ReLU"]
        H2["Neuron H2 Σ w*x + b → ReLU"]
        H3["Neuron H3 Σ w*x + b → ReLU"]
        H4["Neuron H4 Σ w*x + b → ReLU"]
        H5["Neuron H5 Σ w*x + b → ReLU"]
        H6["Neuron H6 Σ w*x + b → ReLU"]
        H7["Neuron H7 Σ w*x + b → ReLU"]
        H8["Neuron H8 Σ w*x + b → ReLU"]
    end

    subgraph "Output Layer"
        O1["Output 1 Δx"]
        O2["Output 2 Δy"]
        Movement["Movement Δx, Δy"]
    end

    %% Input to Hidden Layer
    I1 -->|w_11| H1
    I1 -->|w_12| H2
    I1 -->|w_13| H3
    I1 -->|w_14| H4
    I1 -->|w_15| H5
    I1 -->|w_16| H6
    I1 -->|w_17| H7
    I1 -->|w_18| H8

    I2 -->|w_21| H1
    I2 -->|w_22| H2
    I2 -->|w_23| H3
    I2 -->|w_24| H4
    I2 -->|w_25| H5
    I2 -->|w_26| H6
    I2 -->|w_27| H7
    I2 -->|w_28| H8

    I3 -->|w_31| H1
    I3 -->|w_32| H2
    I3 -->|w_33| H3
    I3 -->|w_34| H4
    I3 -->|w_35| H5
    I3 -->|w_36| H6
    I3 -->|w_37| H7
    I3 -->|w_38| H8

    I4 -->|w_41| H1
    I4 -->|w_42| H2
    I4 -->|w_43| H3
    I4 -->|w_44| H4
    I4 -->|w_45| H5
    I4 -->|w_46| H6
    I4 -->|w_47| H7
    I4 -->|w_48| H8

    %% Hidden Layer to Output Layer (FULLY CONNECTED)
    H1 -->|w_ho1| O1
    H1 -->|w_ho1'| O2
    H2 -->|w_ho2| O1
    H2 -->|w_ho2'| O2
    H3 -->|w_ho3| O1
    H3 -->|w_ho3'| O2
    H4 -->|w_ho4| O1
    H4 -->|w_ho4'| O2
    H5 -->|w_ho5| O1
    H5 -->|w_ho5'| O2
    H6 -->|w_ho6| O1
    H6 -->|w_ho6'| O2
    H7 -->|w_ho7| O1
    H7 -->|w_ho7'| O2
    H8 -->|w_ho8| O1
    H8 -->|w_ho8'| O2

    O1 --> Movement
    O2 --> Movement
```
---

## 🧩 Code snippets (from this repo)

### 1) Save/Load compact binary weights

```zig
// src/nn/base.zig  — save()
pub fn save(self: *NeuralNetwork, path: []const u8) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    const writer = file.writer();
    try writer.writeAll(std.mem.asBytes(&self.weights_ih));
    try writer.writeAll(std.mem.asBytes(&self.weights_ho));
    try writer.writeAll(std.mem.asBytes(&self.bias_h));
    try writer.writeAll(std.mem.asBytes(&self.bias_o));
}

// src/nn/base.zig  — load()
pub fn load(path: []const u8) !NeuralNetwork {
    const file = try std.fs.cwd().openFile(path, .{});
    const reader = file.reader();

    var nn = NeuralNetwork{
        .weights_ih = undefined,
        .weights_ho = undefined,
        .bias_h = undefined,
        .bias_o = undefined,
    };

    try reader.readNoEof(std.mem.asBytes(&nn.weights_ih));
    try reader.readNoEof(std.mem.asBytes(&nn.weights_ho));
    try reader.readNoEof(std.mem.asBytes(&nn.bias_h));
    try reader.readNoEof(std.mem.asBytes(&nn.bias_o));

    return nn;
}
```

### 2) Initialization with explicit randomness (excerpt)

```zig
// src/nn/base.zig — init() excerpt
pub fn init() !NeuralNetwork {
    const range: f32 = 0.1;
    var nn = NeuralNetwork{
        .weights_ih = undefined,
        .weights_ho = undefined,
        .bias_h = undefined,
        .bias_o = undefined,
    };

    // Input → Hidden
    for (0..HiddenSize) |h| {
        for (0..InputSize) |i| {
            var r: u32 = 0;
            std.crypto.random.bytes(std.mem.asBytes(&r));
            nn.weights_ih[h][i] = @as(f32, @floatFromInt(r % 100)) / 100.0 - range;
        }
    }

    // Hidden → Output
    for (0..OutputSize) |o| {
        for (0..HiddenSize) |h| {
            var r: u32 = 0;
            std.crypto.random.bytes(std.mem.asBytes(&r));
            nn.weights_ho[o][h] = @as(f32, @floatFromInt(r % 100)) / 100.0 - range;
        }
    }

    // Biases
    for (0..HiddenSize) |h| {
        var r: u32 = 0;
        std.crypto.random.bytes(std.mem.asBytes(&r));
        nn.bias_h[h] = @as(f32, @floatFromInt(r % 100)) / 100.0 - range;
    }
    for (0..OutputSize) |o| {
        var r: u32 = 0;
        std.crypto.random.bytes(std.mem.asBytes(&r));
        nn.bias_o[o] = @as(f32, @floatFromInt(r % 100)) / 100.0 - range;
    }
    return nn;
}
```

### 3) Forward pass + ReLU (excerpt)

```zig
// src/nn/base.zig — forward path (excerpt)
var hidden: [HiddenSize]f32 = undefined;
var output: [OutputSize]f32 = undefined;

// Hidden layer (ReLU)
for (0..HiddenSize) |h| {
    var sum: f32 = self.bias_h[h];
    for (0..InputSize) |i| {
        sum += self.weights_ih[h][i] * input[i];
    }
    hidden[h] = if (sum > 0) sum else 0; // ReLU
}

// Output layer (linear)
for (0..OutputSize) |o| {
    var sum: f32 = self.bias_o[o];
    for (0..HiddenSize) |h| {
        sum += self.weights_ho[o][h] * hidden[h];
    }
    output[o] = sum; // (Softmax/Sigmoid can be applied by the caller if needed)
}
```

> More code (training loop, gradient updates) is in `src/nn/base.zig`. The full repo includes a live GLFW visualizer.

---

## 🔌 Interop & data

* **Binary weights** via `save()` / `load()`
* **CSV adapters + metrics** (planned MR)
* Optional **Python bridge** for dataset prep (Pandas → normalized CSV)

---

## 📈 Roadmap / Planned MRs

* He/Xavier init (seeded `DefaultPrng`)
* L2 weight decay (bias‑free)
* Early stopping + LR decay
* Dropout (training‑only)
* CSV loader + tiny metrics
* Reproducible example dataset

---

## 📬 Contact & full repo

* Public demo: this repository
* Full version: available **on request**
* **Zbyněk Másler** — [zbynekmasler@gmail.com](mailto:zbynekmasler@gmail.com)

---

## 📝 License

MIT (final license confirmed in the full repo)
