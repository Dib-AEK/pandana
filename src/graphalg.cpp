#include "graphalg.h"
#include <math.h>
#include <algorithm>
#include <limits>

namespace MTC {
namespace accessibility {
Graphalg::Graphalg(
        int numnodes, vector< vector<long> > edges, vector<double> edgeweights,
        bool twoway) {
    this->numnodes = numnodes;

    int num = omp_get_max_threads();
    
    FILE_LOG(logINFO) << "Generating contraction hierarchies with "
                      << num << " threads.\n";
    
    ch = CH::ContractionHierarchies(num);

    vector<CH::Node> nv;

    for (int i = 0 ; i < numnodes ; i++) {
        // CH allows you to pass in a node id, and an x and a y, and then
        // never uses it - to be clear, we don't pass it in anymore
        CH::Node n(i, 0, 0);
        nv.push_back(n);
    }

    FILE_LOG(logINFO) << "Setting CH node vector of size "
                      << nv.size() << "\n";
	
    ch.SetNodeVector(nv);

    // Calculate dynamic multiplier based on edge weight range
    // to avoid overflow with large weights and zero weights with small values
    double minWeight = std::numeric_limits<double>::max();
    double maxWeight = 0.0;
    
    for (int i = 0; i < edgeweights.size(); i++) {
        if (edgeweights[i] > 0) {  // Only consider positive weights
            minWeight = std::min(minWeight, edgeweights[i]);
            maxWeight = std::max(maxWeight, edgeweights[i]);
        }
    }
    
    // Handle edge case where all weights are zero or negative
    if (minWeight == std::numeric_limits<double>::max() || maxWeight == 0.0) {
        distanceMultFact = 1000.0;  // Default multiplier
    } else {
        // Calculate multiplier to ensure:
        // 1. Minimum weight * multiplier >= 1 (to avoid rounding to zero)
        // 2. Maximum weight * multiplier < UINT_MAX (to avoid overflow)
        const double UINT_MAX_SAFE = static_cast<double>(std::numeric_limits<unsigned int>::max()) * 0.9;  // 90% of UINT_MAX for safety
        
        double multForMin = 1.0 / minWeight;  // To make minimum weight at least 1
        double multForMax = UINT_MAX_SAFE / maxWeight;  // To avoid overflow
        
        // Use the smaller of the two to satisfy both constraints
        distanceMultFact = std::min(multForMin, multForMax);
        
        // Ensure multiplier is at least 1.0 to maintain some precision
        distanceMultFact = std::max(distanceMultFact, 1.0);
    }
    
    FILE_LOG(logINFO) << "Using distance multiplier: " << distanceMultFact 
                      << " (min weight: " << minWeight << ", max weight: " << maxWeight << ")\n";

    vector<CH::Edge> ev;

    for (int i = 0 ; i < edges.size() ; i++) {
        CH::Edge e(edges[i][0], edges[i][1], i,
            static_cast<unsigned int>(round(edgeweights[i]*distanceMultFact)), true, twoway);
        ev.push_back(e);
    }

    FILE_LOG(logINFO) << "Setting CH edge vector of size "
                      << ev.size() << "\n";
    
    ch.SetEdgeVector(ev);
    ch.RunPreprocessing();
}


std::vector<NodeID> Graphalg::Route(int src, int tgt, int threadNum) {
    std::vector<NodeID> ResultingPath;

    CH::Node src_node(src, 0, 0);
    CH::Node tgt_node(tgt, 0, 0);

    ch.computeShortestPath(
        src_node,
        tgt_node,
        ResultingPath,
        threadNum);

    return ResultingPath;
}


double Graphalg::Distance(int src, int tgt, int threadNum) {
    CH::Node src_node(src, 0, 0);
    CH::Node tgt_node(tgt, 0, 0);

    unsigned int length = ch.computeLengthofShortestPath(
        src_node,
        tgt_node,
        threadNum);

    return static_cast<double>(length) / distanceMultFact;
}


void Graphalg::Range(int src, double maxdist, int threadNum,
                     DistanceVec &ResultingNodes) {
    CH::Node src_node(src, 0, 0);

    std::vector<std::pair<NodeID, unsigned> > tmp;

    ch.computeReachableNodesWithin(
        src_node,
        maxdist*distanceMultFact,
        tmp,
        threadNum);

    for (int i = 0 ; i < tmp.size() ; i++) {
        std::pair<NodeID, float> node;
        node.first = tmp[i].first;
        node.second = tmp[i].second/distanceMultFact;
        ResultingNodes.push_back(node);
    }
}


DistanceMap
Graphalg::NearestPOI(const POIKeyType &category, int src, double maxdist, int number,
                     int threadNum) {
    DistanceMap dm;

    std::vector<CH::BucketEntry> ResultingNodes;
    ch.getNearestWithUpperBoundOnDistanceAndLocations(
        category,
        src,
        maxdist*distanceMultFact,
        number,
        ResultingNodes,
        threadNum);

    for (int i = 0 ; i < ResultingNodes.size() ; i++) {
        dm[ResultingNodes[i].node] =
            static_cast<float>(ResultingNodes[i].distance) / distanceMultFact;
    }

    return dm;
}
}  // namespace accessibility
}  // namespace MTC
