#pragma once

/// Log information
template <typename _Ty>
static inline void LogInfo(_Ty val)
{
	std::cout << val << std::endl;
}

/// Log information
template <typename _Ty>
static inline void LogInfo(const std::string title, _Ty val)
{
	std::cout << title << ": " << val << std::endl;
}

/// Math, pow2
static inline double pow2(const double t)
{
	return t * t;
}

/// Math, log2
static inline double log2(const size_t n)
{
	return log(n) / log(2);
}

/// Math, logcnk
static inline double logcnk(const size_t n, size_t k)
{
	k = k < n - k ? k : n - k;
	double res = 0;
	for (auto i = 1; i <= k; i++) res += log(double(n - k + i) / i);
	return res;
}

/// Make the vector to a min-heap.
inline void make_min_heap(FRset& vec)
{
	// Min heap
	const auto size = vec.size();
	if (2 <= size)
	{
		for (auto hole = (size + 1) / 2; hole--;)
		{
			const auto val = vec[hole];
			size_t i, child;
			for (i = hole; i * 2 + 1 < size; i = child)
			{
				// Find smaller child
				child = i * 2 + 2;
				if (child == size || vec[child - 1] < vec[child])
				{
					// One child only or the left child is smaller than the right one
					--child;
				}

				// Percolate one level
				if (vec[child] < val)
				{
					vec[i] = vec[child];
				}
				else
				{
					break;
				}
			}
			vec[i] = val;
		}
	}
}

/// Replace the value for the first element and down-heap this element.
inline void min_heap_replace_min_value(FRset& vec, const size_t& val)
{
	// Increase the value of the first element
	const auto size = vec.size();
	size_t i, child;
	for (i = 0; i * 2 + 1 < size; i = child)
	{
		// Find smaller child
		child = i * 2 + 2;
		if (child == size || vec[child - 1] < vec[child])
		{
			// One child only or the left child is smaller than the right one
			--child;
		}

		// Percolate one level
		if (vec[child] < val)
		{
			vec[i] = vec[child];
		}
		else
		{
			break;
		}
	}
	vec[i] = val;
}

/// Make the vector to a max-heap.
static inline void make_max_heap(std::vector<std::pair<float, uint32_t>>& vec)
{
	// Max heap
	const auto size = vec.size();
	if (2 <= size)
	{
		for (auto hole = (size + 1) / 2; hole--;)
		{
			const auto val = vec[hole];
			size_t i, child;
			for (i = hole; i * 2 + 1 < size; i = child)
			{
				// Find smaller child
				child = i * 2 + 2;
				if (child == size || vec[child - 1] > vec[child])
				{
					// One child only or the left child is greater than the right one
					--child;
				}

				// Percolate one level
				if (vec[child] > val)
				{
					vec[i] = vec[child];
				}
				else
				{
					break;
				}
			}
			vec[i] = val;
		}
	}
}

/// Replace the value for the first element and down-heap this element.
static inline void max_heap_replace_max_value(std::vector<std::pair<float, uint32_t>>& vec, const float& val)
{
	// Increase the value of the first element
	const auto size = vec.size();
	size_t i, child;
	auto hole = vec[0];
	for (i = 0; i * 2 + 1 < size; i = child)
	{
		// Find smaller child
		child = i * 2 + 2;
		if (child == size || vec[child - 1] > vec[child])
		{
			// One child only or the left child is greater than the right one
			--child;
		}

		// Percolate one level
		if (vec[child].first > val)
		{
			vec[i] = vec[child];
		}
		else
		{
			break;
		}
	}
	hole.first = val;
	vec[i] = hole;
}

/// Generate one node with probabilities according to their weights for the LT cascade model
static inline size_t gen_random_node_by_weight_LT(const Edgelist& edges)
{
	const double weight = dsfmt_gv_genrand_open_close();//阈值
	size_t minIdx = 0, maxIdx = edges.size() - 1;
	if (weight < edges.front().second) return 0; // 返回First element
	if (weight > edges.back().second) return edges.size() + 1; // No element
	while (maxIdx > minIdx)
	{
		const size_t meanIdx = (minIdx + maxIdx) / 2;
		const auto meanWeight = edges[meanIdx].second;
		if (weight <= meanWeight) maxIdx = meanIdx;
		else minIdx = meanIdx + 1;
	}
	return maxIdx;
}

static double solveQuadratic(double a, double b, double c) 
{
    // 检查系数a是否为零，以避免除以零的情况
    if (a == 0) {
		throw std::runtime_error("Error: 不是一元二次方程，a不能为零");
    }

    double discriminant = b * b - 4 * a * c;

    if (discriminant > 0) {
        // 有两个实数解
        double x1 = (-b + std::sqrt(discriminant)) / (2 * a);
        double x2 = (-b - std::sqrt(discriminant)) / (2 * a);
		//std::cout << "方程有两个实数解：x1 = " << x1 << ", x2 = " << x2 << std::endl;
		if ((x1>0) && (x1<1))
		{
			return x1;
		}else if((x2>0) && (x2<1)){
			return x2;
		}else {
            throw std::runtime_error("Error: 实数解不在指定范围内");
        }

    } else if (discriminant == 0) {
        // 有一个实数解
        double x = -b / (2 * a);
        //std::cout << "方程有一个实数解：x = " << x << std::endl;
		return x;
    } else {
        // 有两个虚数解
        //double realPart = -b / (2 * a);
        //double imaginaryPart = std::sqrt(-discriminant) / (2 * a);
        //std::cout << "方程有两个虚数解：" << std::endl;
        //std::cout << "x1 = " << realPart << " + " << imaginaryPart << "i" << std::endl;
        //std::cout << "x2 = " << realPart << " - " << imaginaryPart << "i" << std::endl;
		throw std::runtime_error("Error: 方程没有实数解");
    }
}

//static 会将函数的链接属性设置为内部链接（internal linkage），这样每个包含该头文件的 .cpp 文件都会有自己的函数副本，不会导致重复定义错误。
static Node choose_node_with_marginGainProbability(const std::map<Node, Inf> &nodeInfMap)
{
	std::vector<Node> nodes;
    std::vector<Inf> probabilities;
	nodes.reserve(nodeInfMap.size());
	probabilities.reserve(nodeInfMap.size());

	Inf sumInf = 0.0;
	for (const auto& pair : nodeInfMap) {
        nodes.push_back(pair.first);
        probabilities.push_back(pair.second); 
		sumInf += pair.second;
    }

	std::vector<double> cumulativeProbabilities;
	cumulativeProbabilities.resize(probabilities.size());
	cumulativeProbabilities[0] = probabilities[0] / sumInf;
	for (size_t i = 1; i < probabilities.size(); ++i) {
		cumulativeProbabilities[i] = cumulativeProbabilities[i - 1] + (probabilities[i] / sumInf);
	}

	double randValue = dsfmt_gv_genrand_close_open();
	std::cout<<randValue<<std::endl;

	// 根据累积概率选择节点
	for (size_t i = 0; i < cumulativeProbabilities.size(); ++i) {
		if (randValue < cumulativeProbabilities[i]) {
			return nodes[i];
		}
	}

	std::cout<<"return Node(INVALID_LAYER, INVALID_NODE)"<<std::endl;
	return Node(INVALID_LAYER, INVALID_NODE);
}
