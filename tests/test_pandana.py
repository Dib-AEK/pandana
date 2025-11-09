import os.path

import numpy as np
import pandas as pd
import pytest

import pandana.network as pdna

from numpy.testing import assert_allclose
from pandas.testing import assert_index_equal

from pandana.testing import skipifci


@pytest.fixture(scope="module")
def sample_osm(request):
    store = pd.HDFStore(os.path.join(os.path.dirname(__file__), "osm_sample.h5"), "r")
    nodes, edges = store.nodes, store.edges

    net = pdna.Network(nodes.x, nodes.y, edges["from"], edges.to, edges[["weight"]])

    net.precompute(2000)

    def fin():
        store.close()

    request.addfinalizer(fin)

    return net


# initialize a second network
@pytest.fixture(scope="module")
def second_sample_osm(request):
    store = pd.HDFStore(os.path.join(os.path.dirname(__file__), "osm_sample.h5"), "r")
    nodes, edges = store.nodes, store.edges
    net = pdna.Network(nodes.x, nodes.y, edges["from"], edges.to, edges[["weight"]])

    net.precompute(2000)

    def fin():
        store.close()

    request.addfinalizer(fin)

    return net


def random_node_ids(net, ssize):
    return pd.Series(np.random.choice(net.node_ids, ssize))


def random_data(ssize):
    return pd.Series(np.random.random(ssize))


def get_connected_nodes(net):
    net.set(pd.Series(net.node_ids))
    s = net.aggregate(10000, type="COUNT")
    # not all the nodes in the sample network are connected
    # get the nodes in the largest connected subgraph
    # from printing the result out I know the largest subgraph has
    # 477 nodes in the sample data
    connected_nodes = s[s == 477].index.values
    return connected_nodes


def random_connected_nodes(net, ssize):
    return pd.Series(np.random.choice(get_connected_nodes(net), ssize))


def random_x_y(sample_osm, ssize):
    bbox = sample_osm.bbox
    x = pd.Series(np.random.uniform(bbox[0], bbox[2], ssize))
    y = pd.Series(np.random.uniform(bbox[1], bbox[3], ssize))
    return x, y


def test_agg_variables_accuracy(sample_osm):
    net = sample_osm

    # test accuracy compared to Pandas functions
    ssize = 50
    r = random_data(ssize)
    connected_nodes = get_connected_nodes(net)
    nodes = random_connected_nodes(net, ssize)
    net.set(nodes, variable=r)

    s = net.aggregate(100000, type="count").loc[connected_nodes]
    assert s.unique().size == 1
    assert s.iloc[0] == 50

    s = net.aggregate(100000, type="AVE").loc[connected_nodes]
    assert s.describe()["std"] < 0.01  # assert almost equal
    assert_allclose(s.mean(), r.mean(), atol=1e-3)

    s = net.aggregate(100000, type="mean").loc[connected_nodes]
    assert s.describe()["std"] < 0.01  # assert almost equal
    assert_allclose(s.mean(), r.mean(), atol=1e-3)

    s = net.aggregate(100000, type="min").loc[connected_nodes]
    assert s.describe()["std"] < 0.01  # assert almost equal
    assert_allclose(s.mean(), r.min(), atol=1e-3)

    s = net.aggregate(100000, type="max").loc[connected_nodes]
    assert s.describe()["std"] < 0.01  # assert almost equal
    assert_allclose(s.mean(), r.max(), atol=1e-3)

    r.sort_values(inplace=True)

    s = net.aggregate(100000, type="median").loc[connected_nodes]
    assert s.describe()["std"] < 0.01  # assert almost equal
    assert_allclose(s.mean(), r.iloc[25], atol=1e-2)

    s = net.aggregate(100000, type="25pct").loc[connected_nodes]
    assert s.describe()["std"] < 0.01  # assert almost equal
    assert_allclose(s.mean(), r.iloc[12], atol=1e-2)

    s = net.aggregate(100000, type="75pct").loc[connected_nodes]
    assert s.describe()["std"] < 0.01  # assert almost equal
    assert_allclose(s.mean(), r.iloc[37], atol=1e-2)

    s = net.aggregate(100000, type="SUM").loc[connected_nodes]
    assert s.describe()["std"] < 0.05  # assert almost equal
    assert_allclose(s.mean(), r.sum(), atol=1e-2)

    s = net.aggregate(100000, type="std").loc[connected_nodes]
    assert s.describe()["std"] < 0.01  # assert almost equal
    assert_allclose(s.mean(), r.std(), atol=1e-2)


def test_non_integer_nodeids(request):

    store = pd.HDFStore(os.path.join(os.path.dirname(__file__), "osm_sample.h5"), "r")
    nodes, edges = store.nodes, store.edges

    # convert to string!
    nodes.index = nodes.index.astype("str")
    edges["from"] = edges["from"].astype("str")
    edges["to"] = edges["to"].astype("str")

    net = pdna.Network(nodes.x, nodes.y, edges["from"], edges.to, edges[["weight"]])

    def fin():
        store.close()

    request.addfinalizer(fin)

    # test accuracy compared to Pandas functions
    ssize = 50
    r = random_data(ssize)
    connected_nodes = get_connected_nodes(net)
    random_nodes = random_connected_nodes(net, ssize)
    net.set(random_nodes, variable=r)

    s = net.aggregate(100000, type="count").loc[connected_nodes]
    assert list(nodes.index), list(s.index)


def test_agg_variables(sample_osm):
    net = sample_osm

    ssize = 50
    net.set(random_node_ids(sample_osm, ssize), variable=random_data(ssize))

    for type in net.aggregations:
        for decay in net.decays:
            for distance in [5, 10, 20]:
                t = type.decode(encoding="UTF-8")
                d = decay.decode(encoding="UTF-8")
                s = net.aggregate(distance, type=t, decay=d)
                assert s.describe()["std"] > 0

    # testing w/o setting variable
    ssize = 50
    net.set(random_node_ids(sample_osm, ssize))

    for type in net.aggregations:
        for decay in net.decays:
            for distance in [5, 10, 20]:
                t = type.decode(encoding="UTF-8")
                d = decay.decode(encoding="UTF-8")
                s = net.aggregate(distance, type=t, decay=d)
                if t != "std":
                    assert s.describe()["std"] > 0
                else:
                    # no variance in data
                    assert s.describe()["std"] == 0


def test_non_float_node_values(sample_osm):
    net = sample_osm

    ssize = 50
    net.set(
        random_node_ids(sample_osm, ssize),
        variable=(random_data(ssize) * 100).astype("int"),
    )

    for type in net.aggregations:
        for decay in net.decays:
            for distance in [5, 10, 20]:
                t = type.decode(encoding="UTF-8")
                d = decay.decode(encoding="UTF-8")
                s = net.aggregate(distance, type=t, decay=d)
                assert s.describe()["std"] > 0


def test_missing_nodeid(sample_osm):
    node_ids = random_node_ids(sample_osm, 50)
    # non-existing value
    node_ids.iloc[0] = -1
    sample_osm.set(node_ids)


def test_assign_nodeids(sample_osm):
    ssize = 50
    np.random.seed(0)
    x, y = random_x_y(sample_osm, ssize)
    node_ids1 = sample_osm.get_node_ids(x, y)
    assert len(node_ids1) == ssize
    # check a couple of assignments for accuracy
    assert node_ids1.loc[48] == 1840703798
    assert node_ids1.loc[43] == 257739973
    assert_index_equal(x.index, node_ids1.index)

    # test with max distance - this max distance is in decimal degrees
    node_ids2 = sample_osm.get_node_ids(x, y, 0.0005)
    assert 0 < len(node_ids2) < ssize
    assert len(node_ids2) < len(node_ids1), "Max distance not working"
    assert len(node_ids2) == 14

    node_ids3 = sample_osm.get_node_ids(x, y, 0)
    assert len(node_ids3) == 0


def test_named_variable(sample_osm):
    net = sample_osm

    ssize = 50
    net.set(random_node_ids(sample_osm, ssize), variable=random_data(ssize), name="foo")

    net.aggregate(500, type="sum", decay="linear", name="foo")


"""
def test_plot(sample_osm):
    net = sample_osm

    ssize = 50
    net.set(random_node_ids(sample_osm, ssize),
            variable=random_data(ssize))

    s = net.aggregate(500, type="sum", decay="linear")

    sample_osm.plot(s)
"""


def test_shortest_path(sample_osm):

    for i in range(10):
        ids = random_connected_nodes(sample_osm, 2)
        path = sample_osm.shortest_path(ids[0], ids[1])
        assert path.size >= 2
        assert ids[0] == path[0]
        assert ids[1] == path[-1]


def test_shortest_paths(sample_osm):

    nodes = random_connected_nodes(sample_osm, 100)
    vec_paths = sample_osm.shortest_paths(nodes[0:50], nodes[50:100])

    for i in range(50):
        path = sample_osm.shortest_path(nodes[i], nodes[i + 50])
        assert np.array_equal(vec_paths[i], path)

    # check mismatched OD lists
    try:
        vec_paths = sample_osm.shortest_paths(nodes[0:51], nodes[50:100])
        assert 0
    except ValueError as e:
        pass


def test_shortest_path_length(sample_osm):

    for i in range(10):
        ids = random_connected_nodes(sample_osm, 2)
        len = sample_osm.shortest_path_length(ids[0], ids[1])
        assert len >= 0


def test_shortest_path_lengths(sample_osm):

    nodes = random_connected_nodes(sample_osm, 100)
    lens = sample_osm.shortest_path_lengths(nodes[0:50], nodes[50:100])
    for len in lens:
        assert len >= 0

    # check mismatched OD lists
    try:
        lens = sample_osm.shortest_path_lengths(nodes[0:51], nodes[50:100])
        assert 0
    except ValueError as e:
        pass


def test_shortest_path_length_precision():
    """
    Test that shortest_path_lengths uses rounding instead of truncation
    when converting edge weights to integers, reducing precision loss.
    
    This test addresses the bug where shortest_path_lengths gave different 
    results than computing the length from shortest_paths due to truncation
    of edge weights (multiplied by 1000 for internal integer representation).
    
    With rounding, the maximum error per edge is 0.0005 (half of 0.001).
    With truncation, it was up to 0.001 per edge.
    """
    # Create a simple test network with non-integer edge weights
    nodes = pd.DataFrame({
        'x': [0.0, 1.0, 2.0, 3.0],
        'y': [0.0, 0.0, 0.0, 0.0]
    }, index=[1, 2, 3, 4])
    
    # Use edge weights that demonstrate the difference between rounding and truncation
    # 0.9999 * 1000 = 999.9, which truncates to 999 but rounds to 1000
    edges = pd.DataFrame({
        'from': [1, 2, 3],
        'to': [2, 3, 4],
        'weight': [0.9999, 1.5005, 2.3007]
    })
    
    net = pdna.Network(nodes.x, nodes.y, edges['from'], edges.to, 
                       edges[['weight']], twoway=False)
    
    # Test single query
    path = net.shortest_path(1, 4)
    reported_length = net.shortest_path_length(1, 4)
    actual_sum = sum(edges['weight'])
    
    # With rounding: (1000 + 1501 + 2301) / 1000 = 4.802
    # With truncation: (999 + 1500 + 2300) / 1000 = 4.799
    # Actual sum: 0.9999 + 1.5005 + 2.3007 = 4.8011
    
    # The reported length should be closer to actual with rounding
    error = abs(reported_length - actual_sum)
    
    # Maximum expected error with rounding: 0.0005 per edge * 3 edges = 0.0015
    # Add small tolerance for floating point comparison
    assert error <= 0.002, f"Error {error} exceeds expected maximum with rounding"
    
    # Verify rounding is used (reported ≈ 4.802) not truncation (reported ≈ 4.799)
    assert abs(reported_length - 4.802) < 0.001, \
        f"Expected rounding behavior (≈4.802), got {reported_length}"
    
    # Test vectorized version
    vec_lengths = net.shortest_path_lengths([1], [4])
    assert abs(vec_lengths[0] - reported_length) < 0.0001, \
        "Vectorized version should give same result as single query"


def test_shortest_path_extreme_impedances():
    """
    Test that shortest path calculations work correctly with very small 
    and very large impedance values.
    
    This addresses the bug where impedance values outside a certain range
    (too small ~<1e-4 or too large ~>1e7) would cause incorrect shortest
    path solutions due to:
    1. Small values rounding to zero after multiplication
    2. Large values overflowing unsigned int after multiplication
    """
    # Create a simple network with two alternative paths
    nodes = pd.DataFrame({
        'x': [0.0, 1.0, 2.0, 3.0, 4.0],
        'y': [0.0, 0.0, 0.0, 0.0, 0.0]
    }, index=[1, 2, 3, 4, 5])
    
    # Path 1->2->3->4 has total weight 0.003 (3 edges @ 0.001 each)
    # Path 1->5->4 has total weight 0.004 (2 edges @ 0.002 each)
    # The correct shortest path should be 1->2->3->4
    edges = pd.DataFrame({
        'from': [1, 2, 3, 1, 5],
        'to': [2, 3, 4, 5, 4],
        'weight': [0.001, 0.001, 0.001, 0.002, 0.002]
    })
    
    # Test 1: Original weights - should work correctly
    net = pdna.Network(nodes.x, nodes.y, edges['from'], edges.to, 
                       edges[['weight']], twoway=False)
    path = net.shortest_path(1, 4)
    assert list(path) == [1, 2, 3, 4], \
        f"Expected path [1, 2, 3, 4] with original weights, got {list(path)}"
    
    # Test 2: Small weights (scaled down by 10) - this was broken before
    edges_small = edges.copy()
    edges_small['weight'] = edges['weight'] / 10
    net_small = pdna.Network(nodes.x, nodes.y, edges_small['from'], 
                             edges_small.to, edges_small[['weight']], twoway=False)
    path_small = net_small.shortest_path(1, 4)
    assert list(path_small) == [1, 2, 3, 4], \
        f"Expected path [1, 2, 3, 4] with small weights (0.0001-0.0002), got {list(path_small)}"
    
    # Verify the path length is also correct
    length_small = net_small.shortest_path_length(1, 4)
    expected_length_small = 0.0001 + 0.0001 + 0.0001  # 0.0003
    assert abs(length_small - expected_length_small) < 0.00001, \
        f"Expected length ~{expected_length_small} with small weights, got {length_small}"
    
    # Test 3: Very small weights (scaled down by 100) - even more extreme
    edges_tiny = edges.copy()
    edges_tiny['weight'] = edges['weight'] / 100
    net_tiny = pdna.Network(nodes.x, nodes.y, edges_tiny['from'], 
                            edges_tiny.to, edges_tiny[['weight']], twoway=False)
    path_tiny = net_tiny.shortest_path(1, 4)
    assert list(path_tiny) == [1, 2, 3, 4], \
        f"Expected path [1, 2, 3, 4] with very small weights (0.00001-0.00002), got {list(path_tiny)}"
    
    # Test 4: Large weights (scaled up significantly) - this was also broken before
    edges_large = edges.copy()
    edges_large['weight'] = edges['weight'] * 1e7
    net_large = pdna.Network(nodes.x, nodes.y, edges_large['from'], 
                             edges_large.to, edges_large[['weight']], twoway=False)
    path_large = net_large.shortest_path(1, 4)
    assert list(path_large) == [1, 2, 3, 4], \
        f"Expected path [1, 2, 3, 4] with large weights (1e4-2e4), got {list(path_large)}"
    
    # Verify the path length is also correct for large weights
    length_large = net_large.shortest_path_length(1, 4)
    expected_length_large = 1e4 + 1e4 + 1e4  # 3e4
    assert abs(length_large - expected_length_large) / expected_length_large < 0.01, \
        f"Expected length ~{expected_length_large} with large weights, got {length_large}"
    
    # Test 5: Very large weights (even more extreme)
    edges_huge = edges.copy()
    edges_huge['weight'] = edges['weight'] * 1e8
    net_huge = pdna.Network(nodes.x, nodes.y, edges_huge['from'], 
                            edges_huge.to, edges_huge[['weight']], twoway=False)
    path_huge = net_huge.shortest_path(1, 4)
    assert list(path_huge) == [1, 2, 3, 4], \
        f"Expected path [1, 2, 3, 4] with very large weights (1e5-2e5), got {list(path_huge)}"
    
    # Test 6: Test vectorized versions with small weights
    paths_vec = net_small.shortest_paths([1], [4])
    assert list(paths_vec[0]) == [1, 2, 3, 4], \
        f"Expected path [1, 2, 3, 4] with vectorized call, got {list(paths_vec[0])}"
    
    lengths_vec = net_small.shortest_path_lengths([1], [4])
    assert abs(lengths_vec[0] - expected_length_small) < 0.00001, \
        f"Expected length ~{expected_length_small} with vectorized call, got {lengths_vec[0]}"


def test_pois(sample_osm):
    net = sample_osm

    ssize = 50
    np.random.seed(0)
    x, y = random_x_y(sample_osm, ssize)

    with pytest.raises(AssertionError):
        net.nearest_pois(2000, "restaurants", num_pois=10)

    with pytest.raises(AssertionError):
        net.nearest_pois(2000, "restaurants", num_pois=10)

    # boundary condition
    net.set_pois("restaurants", 2000, 10, x, y)

    net.nearest_pois(2000, "restaurants", num_pois=10)

    with pytest.raises(AssertionError):
        net.nearest_pois(2000, "restaurants", num_pois=11)

    net = sample_osm
    x, y = random_x_y(sample_osm, 100)
    x.index = ["lab%d" % i for i in range(len(x))]
    y.index = x.index

    net.set_pois("restaurants", 2000, 10, x, y)

    d = net.nearest_pois(2000, "restaurants", num_pois=10, include_poi_ids=True)


def test_pois2(second_sample_osm):
    net2 = second_sample_osm

    ssize = 50
    np.random.seed(0)
    x, y = random_x_y(second_sample_osm, ssize)

    # make sure POI searches work on second graph
    net2.set_pois("restaurants", 2000, 10, x, y)

    net2.nearest_pois(2000, "restaurants", num_pois=10)


def test_pois_pandana3(second_sample_osm):
    net2 = second_sample_osm

    ssize = 50
    np.random.seed(0)
    x, y = random_x_y(second_sample_osm, ssize)
    pdna.reserve_num_graphs(1)

    net2.init_pois(num_categories=1, max_dist=2000, max_pois=10)

    # make sure POI searches work on second graph
    net2.set_pois(category="restaurants", x_col=x, y_col=y)

    net2.nearest_pois(2000, "restaurants", num_pois=10)


def test_pois_pandana3_pos_args(second_sample_osm):
    net2 = second_sample_osm

    ssize = 50
    np.random.seed(0)
    x, y = random_x_y(second_sample_osm, ssize)
    pdna.reserve_num_graphs(1)

    net2.init_pois(1, 2000, 10)

    # make sure poi searches work on second graph
    net2.set_pois("restaurants", x, y)

    net2.nearest_pois(2000, "restaurants", num_pois=10)


# test items are sorted


def test_sorted_pois(sample_osm):
    net = sample_osm

    ssize = 1000
    x, y = random_x_y(sample_osm, ssize)

    # set two categories
    net.set_pois("restaurants", 2000, 10, x, y)

    test = net.nearest_pois(2000, "restaurants", num_pois=10)

    for ind, row in test.iterrows():
        # make sure it's sorted
        assert_allclose(row, row.sort_values())


def test_repeat_pois(sample_osm):
    net = sample_osm

    def get_nearest_nodes(x, y, x2=None, y2=None, n=2):
        coords_dict = [{"x": x, "y": y, "var": 1} for i in range(2)]
        if x2 and y2:
            coords_dict.append({"x": x2, "y": y2, "var": 1})
        df = pd.DataFrame(coords_dict)
        sample_osm.set_pois("restaurants", 2000, 10, df["x"], df["y"])
        res = sample_osm.nearest_pois(
            2000, "restaurants", num_pois=5, include_poi_ids=True
        )
        return res

    # these are the min-max values of the network
    # -122.3383688 -122.2962223
    # 47.5950005 47.6150548

    test1 = get_nearest_nodes(-122.31, 47.60)
    test2 = get_nearest_nodes(-122.254116, 37.869361)
    # Same coords as the first call, should yield same result
    test3 = get_nearest_nodes(-122.31, 47.60)
    assert test1.equals(test3)

    test4 = get_nearest_nodes(-122.31, 47.60, -122.32, 47.61, n=3)
    assert_allclose(
        test4.loc[53114882], [7, 13, 13, 2000, 2000, 2, 0, 1, np.nan, np.nan]
    )
    assert_allclose(
        test4.loc[53114880], [6, 14, 14, 2000, 2000, 2, 0, 1, np.nan, np.nan]
    )
    assert_allclose(
        test4.loc[53227769],
        [2000, 2000, 2000, 2000, 2000, np.nan, np.nan, np.nan, np.nan, np.nan],
    )


def test_nodes_in_range(sample_osm):
    net = sample_osm

    np.random.seed(0)
    ssize = 10
    x, y = random_x_y(net, 10)
    snaps = net.get_node_ids(x, y)

    test1 = net.nodes_in_range(snaps, 1)
    net.precompute(10)
    test5 = net.nodes_in_range(snaps, 5)
    test11 = net.nodes_in_range(snaps, 11)
    assert test1.weight.max() == 1
    assert test5.weight.max() == 5
    assert test11.weight.max() == 11

    focus_id = snaps[0]
    all_distances = net.shortest_path_lengths(
        [focus_id] * len(net.node_ids), net.node_ids
    )
    all_distances = np.asarray(all_distances)
    assert (all_distances <= 1).sum() == len(
        test1.query("source == {}".format(focus_id))
    )
    assert (all_distances <= 5).sum() == len(
        test5.query("source == {}".format(focus_id))
    )
    assert (all_distances <= 11).sum() == len(
        test11.query("source == {}".format(focus_id))
    )
