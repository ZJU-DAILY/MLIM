#pragma once

/// node id
typedef uint32_t NodeID;
///layer id
typedef uint32_t LayerID;
///degree of coverage
typedef uint32_t CoverDegree;
/// represent a node
typedef std::pair<LayerID, NodeID> Node;
/// Node list
typedef std::vector<Node> Nodelist;
/// Edge structure, neighbor id and the edge weight，这里的Node是起点
typedef std::pair<Node, float> Edge;
/// Edgelist structure from one source/target node
typedef std::vector<Edge> Edgelist;
/// One forward reachable set
typedef std::vector<size_t> FRset;
/// A set of forward reachable sets
typedef std::vector<FRset> FRsets;
/// One reverse reachable set
typedef struct reverseReachableSet
{
	std::vector<Node> rrSetContent;//RR set的内容
	LayerID sourceLayer;//这条RR set的采样的源节点所属的层
}RRset;
/// A set of reverse reachable sets
typedef std::vector<RRset> RRsets;
/// Coverage List
typedef std::map<Node, CoverDegree> Coverage;
/// CoverageDegreeMap
typedef std::vector<Nodelist> CoverageDegreeMap;
/// marin gain
typedef double Inf;

// 使用 constexpr 定义常量。 constexpr 可以检查类型是否匹配，还可以使得编译器会确保常量在每个源文件中只定义一次，从而避免链接冲突
constexpr LayerID INVALID_LAYER = -1;  // 定义非法的 LayerID
constexpr NodeID INVALID_NODE = -1;    // 定义非法的 NodeID


/// Cascade models: IC, LT
enum CascadeModel { IC = 0, LT = 1, DLT = 2 };

/// Node element with id and a property value
typedef struct NodeElement
{
	int id;
	double value;
} NodeEleType;

/// Smaller operation for node element
struct smaller
{
	bool operator()(const NodeEleType& Ele1, const NodeEleType& Ele2) const
	{
		return (Ele1.value < Ele2.value);
	}
};

/// Greater operation for node element
struct greater
{
	bool operator()(const NodeEleType& Ele1, const NodeEleType& Ele2) const
	{
		return (Ele1.value > Ele2.value);
	}
};
