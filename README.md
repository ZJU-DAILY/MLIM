# MLIM

This is our C++ & Python implementation for the paper:
> Xueqin Chang, Ruize Liu, Qing Liu. Theoretical and Learning-based Approaches for Influence Maximization in Multilayer Social Networks. 2025.


## Environment Requirements

### STARIM
For STARIM, we implement the algorithm in C++, and the detailed environment requirements are as follows.
The project is developed and tested under a Linux environment.
We recommend the following configuration for stable compilation and execution:
| Component            | Recommended Version                                        | Description                                          |
| -------------------- | ---------------------------------------------------------- | ---------------------------------------------------- |
| **Operating System** | Ubuntu 20.04 / 22.04 (or other modern Linux distributions) | Verified under Ubuntu                                |
| **Compiler**         | g++ ≥ 7.5.0                                                | Must support the C++14 standard                      |
| **Build Tool**       | make (optional)                                            | If you prefer to use Makefile for compilation        |
| **Memory**           | ≥ 16 GB RAM                                                | Large-scale graph processing may require more memory |
| **Dependencies**     | None                                                       | Only requires a standard C++14 environment           |


### LGQIM

For LGQIM, we implement it using Python. The specific environment requirements are as follows.
We strongly recommend using Anaconda to create and manage the Python environment required for running this project.

You can create a new environment as follows:

  ```
  conda create -n LGQIM python=3.9
  conda activate LGQIM
  ```

#### Recommended Versions
| Component              | Recommended Version     | Description                          |
| ---------------------- | ----------------------- | ------------------------------------ |
| **Python**             | ≥ 3.9 (tested with 3.9) | Python 3.9 or higher is recommended  |
| **PyTorch**            | 2.5.1                   | Tested with `torch==2.5.1`           |
| **CUDA (for GPU use)** | 12.1                    | Verified to work with `torch==2.5.1` |
| **cuDNN**              | 7.4.2                   | Verified to work with CUDA 12.1      |

#### Required Python Packages
You can install all dependencies with pip or conda.
Below are the essential and auxiliary libraries:

Essential libraries
torch, networkx, scikit-learn, argparse, subprocess

Auxiliary libraries
time, random, sys, os, re

⚙️ CPU and GPU Support
This code can run on both CPU and GPU:

CPU:
Simply install the CPU version of PyTorch using:
pip install torch==2.5.1

GPU:
Install the GPU-enabled version of PyTorch (recommended CUDA 12.1 and cuDNN 7.4.2).
Both versions have been verified to work correctly in our experiments.

## Dataset

### Dataset Statistics
The following datasets are used in our experiments.
Each dataset is represented as a multilayer or multiplex network, characterized by the number of nodes, number of edges, and number of layers.
| Dataset             | Abbreviation | #Nodes    | #Edges     | #Layers |
| ------------------- | ------------ | --------- | ---------- | ------- |
| Venetie             | **VT**       | 1,380     | 19,941     | 43      |
| FF-TW-YT            | **FTY**      | 11,863    | 87,396     | 3       |
| Citeseer            | **CS**       | 15,533    | 122,903    | 3       |
| DBLP                | **DBLP**     | 41,892    | 661,883    | 2       |
| Twitter             | **TW**       | 47,807    | 657,456    | 3       |
| MoscowAthletics2013 | **MA**       | 133,364   | 309,952    | 3       |
| Frienfeed           | **FF**       | 1,531,015 | 20,204,535 | 3       |
| ObamaInIsrael2013   | **OB**       | 3,452,114 | 6,711,448  | 3       |
| SF                  | **SF**       | 4,975,888 | 63,497,050 | 9       |

Due to the size limitation of the datasets, only three datasets have been placed in this repository as examples. They are: Citeseer, DBLP, and ff-tw-yt, located under the dataset folder of STARIM.

### Dataset Format
Each dataset is organized as follows:

#### 1. **Layer Files**

For each layer, three files are provided:
`layer{x}.txt`, `layer{x}model.txt`, and `layer{x}ov.txt`

These files respectively store:

* the **intra-layer edges**,
* the **propagation model** of the layer, and
* the **inter-layer (overlapping) edges**.

Here, `{x}` is replaced by the corresponding layer index.
For example, the **Citeseer** dataset contains three layers, and the related files are:

```
layer1.txt, layer1model.txt, layer1ov.txt  
layer2.txt, layer2model.txt, layer2ov.txt  
layer3.txt, layer3model.txt, layer3ov.txt
```

---

#####  File: `layer{x}.txt`

* **Format:**

  * The **first line** contains two integers:

    ```
    <number_of_nodes> <number_of_edges>
    ```
  * Starting from the **second line**, each line represents one directed edge:

    ```
    src dst weight
    ```
  * `weight` ∈ [0, 1] denotes the influence strength of the edge from `src` to `dst` on the *x*-th layer.

---

#####  File: `layer{x}model.txt`

* Stores the **propagation model type** for the *x*-th layer.
* Encoding:

  * `0` → Independent Cascade (IC) model
  * `1` → Linear Threshold (LT) model

---

#####  File: `layer{x}ov.txt`

* Records the **cross-layer (overlap) edges**.
* **Format:**

  ```
  node_id overlap_layer_id overlap_node_id weight
  ```

  This represents an inter-layer edge from
  `(overlap_layer_id, overlap_node_id)` → `(x, node_id)`
  with weight `weight`.
  In other words, a node `overlap_node_id` in layer `overlap_layer_id` influences node `node_id` in layer *x*.

---

#### 2. **max_nodeID.txt**

Stores the **maximum node ID** in the dataset.

---

#### 3. **node_score.txt**

Stores the **initial influence score** of each node, which is used for training the learning-based methods.

---

#### 4. **total_layers.txt**

Stores the **total number of layers** in the dataset.


## Reproducibility & Run

###  Running STARIM
To run **STARIM**, navigate to the `STARIM` directory and compile the code using the following command:

```bash
g++ -std=c++14 -g -O3 ./MGRR.cpp ./multiplex.cpp -o executableFile/STARIM
```

---

####  Example of Execution

As an example, you can run STARIM on the **Citeseer** dataset with the following command:

```bash
./executableFile/STARIM -mode=M -dir=dataset/Citeseer -seedsize=20 -delta=0.01 -epsilon=0.2
```

---

####  Additional Implementations

In this project, we also provide implementations of several **baseline methods** and **utility tools** used in the experiments.
Their compilation and execution processes are similar — simply replace `./MGRR.cpp` with the corresponding `.cpp` source file, and then compile and run it in the same way.


###  Running LGQIM

To run **LGQIM**, please navigate to the `LGQIM` directory.
LGQIM consists of **two main components**: **GNN training** and **QNet training**.

---

####  1. GNN Training

You can train the GNN model using the following command:

```bash
python GNN.py --dataset "$dataset" --num_epochs "$r"
```

where:

* `"$dataset"` should be replaced with the **path to the dataset**,
* `"$r"` should be replaced with the **number of training epochs** for the GNN model.

---

####  2. QNet Training

You can train the QNet model using:

```bash
python QNet_model/QNet_train.py --qnet_dataset "$dataset" --qnet_k "$k" --qnet_gamma "$gm"
```

where:

* `"$l_dataset"` — path to the dataset used for QNet training.
* `"$k"` — number of seed nodes to be selected.
* `"$gm"` — parameter **gamma (γ)** in QNet, representing the *discount factor* for Q-learning.

  * Optional values: `[0, 0.2, 0.4, 0.6, 0.8, 1.0]`
  * Larger `gamma` implies higher influence.
  * The typical setting is `0.8`.




