# MLIM

This is our implementation for the paper:
> Approximation and Learning-based Algorithms for Influence Maximization in Multilayer Social Networks.


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

Due to the size limitation of the datasets, only three datasets have been placed in this repository as examples. They are: Citeseer, DBLP, and ff-tw-yt.

### Dataset Sources and References

The datasets used in this project are collected from publicly available sources. Their original references are listed as follows. If you use these datasets, please cite the corresponding papers.

* **Venetie**, **Twitter**
  
  J. A. Baggio, S. B. BurnSilver, A. Arenas, J. S. Magdanz, G. P. Kofinas, and M. De Domenico,
  *Multiplex social ecological network analysis reveals how social changes affect community robustness more than resource depletion*,
  **Proceedings of the National Academy of Sciences**, vol. 113, no. 48, pp. 13708–13713, 2016.

* **FF-TW-YT**, **Friendfeed**
  
  M. E. Dickison, M. Magnani, and L. Rossi,
  *Multilayer Social Networks*,
  Cambridge University Press, 2016.

* **Citeseer**, **DBLP**
  
  D. Liu and Z. Zou,
  *Gcore: Exploring cross-layer cohesiveness in multilayer graphs*,
  **Proceedings of the VLDB Endowment**, vol. 16, no. 11, pp. 3201–3213, 2023.

* **MoscowAth**, **Obama**
  
  E. Omodei, M. De Domenico, and A. Arenas,
  *Characterizing interactions in online social networks during exceptional events*,
  **Frontiers in Physics**, vol. 3, p. 59, 2015.

* **StackOverflow**
  
  J. Leskovec and A. Krevl,
  *SNAP Datasets: Stanford Large Network Dataset Collection*, 2014. Available at: [http://snap.stanford.edu/data](http://snap.stanford.edu/data)

---

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
python QNet_model/QNet_train.py --qnet_dataset "$l_dataset" --qnet_k "$k" --qnet_gamma "$gm"
```

where:

* `"$l_dataset"` — path to the dataset used for QNet training.
* `"$k"` — number of seed nodes to be selected.
* `"$gm"` — parameter **gamma (γ)** in QNet, representing the *discount factor* for Q-learning.

  * Optional values: `[0, 0.2, 0.4, 0.6, 0.8, 1.0]`
  * Larger `gamma` implies higher influence.
  * The typical setting is `0.8`.












