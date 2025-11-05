# Copyright 2025 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Any
import warnings

import networkx as nx

import dimod
import numpy as np

from dimod import ComposedSampler

try:
    from dwave.experimental.automorphism import schreier_rep
    from dwave.experimental.automorphism import (
        sample_automorphisms as sample_automorphisms_u_vectors,
    )
except:
    warnings.warn(
        "dwave experimental unavailable, use of u_vectors and "
        "generator automation is not available. See "
        "dwave-experimental"
    )

__all__ = [
    "AutomorphismComposite",
    "chimera_generators",
    "pegasus_generators",
    "zephyr_generators",
    "sample_automorphisms_listdict",
]


def chimera_generators(m: int, n: Optional[int] = None, t: int = 4) -> list[dict]:
    if n is None:
        n = m

    if n == m:
        diagonal = (
            {
                (i, j, u, k): (i, j, 1 - u, k)
                for i in range(m)
                for j in range(n)
                for u in range(2)
                for k in range(t)
            },
            2,
        )
    vertical = (
        {
            (i, j, u, k): (m - 1 - i, j, u, k)
            for i in range(m)
            for j in range(n)
            for u in range(2)
            for k in range(t)
        },
        2,
    )
    horizontal = (
        {
            (i, j, u, k): (i, n - 1 - j, u, k)
            for i in range(m)
            for j in range(n)
            for u in range(2)
            for k in range(t)
        },
        2,
    )
    # Shore Permutations correlated by row/column
    # - must be ordered: (1,..,k) then (1,,k-1) .. (1,2) etc. fully mixes (permutation)
    vert_shores = [
        (
            {
                (i, j, 0, k): (i, j, 0, (k + 1) % kp)
                for i in range(m)
                for k in range(kp)
            },
            kp,
        )
        for kp in range(t, 1, -1)
        for j in range(n)
    ]
    horiz_shores = [
        (
            {
                (i, j, 1, k): (i, j, 1, (k + 1) % kp)
                for j in range(n)
                for k in range(kp)
            },
            kp,
        )
        for kp in range(t, 1, -1)
        for i in range(m)
    ]
    if n == m:
        return [diagonal, vertical, horizontal] + vert_shores + horiz_shores
    else:
        return [vertical, horizontal] + vert_shores + horiz_shores


def flipZephyr(u: int, w: int, k: int, j: int, z: int, orient: bool, m: int) -> tuple:
    if orient == True:
        ue = u
    else:
        ue = 1 - u
    return (
        u,
        ue * (2 * m - w) + (1 - ue) * w,
        k,
        ue * j + (1 - ue) * (1 - j),
        ue * z + (1 - ue) * (m - 1 - z),
    )


def zephyr_generators(m: int, t: int = 4) -> list[dict]:
    """Create generators for zephyr

    m : int
        Grid parameter for the Zephyr lattice.
    t : int
        Tile parameter for the Zephyr lattice.
    """

    diagonal = (
        {
            (u, w, k, j, z): (1 - u, w, k, j, z)
            for u in range(2)
            for w in range(2 * m + 1)
            for k in range(t)
            for j in range(2)
            for z in range(m)
        },
        2,
    )
    vertical = (
        {
            (u, w, k, j, z): flipZephyr(u, w, k, j, z, True, m)
            for u in range(2)
            for w in range(2 * m + 1)
            for k in range(t)
            for j in range(2)
            for z in range(m)
        },
        2,
    )
    horizontal = (
        {
            (u, w, k, j, z): flipZephyr(u, w, k, j, z, False, m)
            for u in range(2)
            for w in range(2 * m + 1)
            for k in range(t)
            for j in range(2)
            for z in range(m)
        },
        2,
    )
    # Shore Permutations correlated by row/column
    # - must be ordered: (1,..,k) then (1,,k-1) .. (1,2) etc. fully mixes (permutation)
    shores = [
        (
            {
                (u, w, k, j, z): (u, w, (k + 1) % kp, j, z)
                for z in range(m)
                for j in range(2)
                for k in range(kp)
            },
            kp,
        )
        for kp in range(t, 1, -1)
        for u in range(2)
        for w in range(2 * m + 1)
    ]

    return [diagonal, vertical, horizontal] + shores


def pegasus_generators(m: int) -> list[dict]:
    """Create generators for pegasus.

    A reflection on the main diagonal, and exchanges of oddly coupled pairs.

    m : int
        Grid parameter for the Pegasus lattice.
    """

    diagonal = (
        {
            (u, w, k, z): (1 - u, m - 1 - w, 11 - k, m - 2 - z)
            for u in range(2)
            for w in range(m)
            for k in range(12)
            for z in range(m - 1)
        },
        2,
    )
    # Odd-pairs
    odd_pairs = [
        (
            {
                (u, w, koff + kin, z): (u, w, koff + (kin + 1) % 2, z)
                for z in range(m - 1)
                for kin in range(2)
            },
            2,
        )
        for koff in range(0, 12, 2)
        for u in range(2)
        for w in range(m)
    ]

    return [diagonal] + odd_pairs


def sample_automorphisms_listdict(generators_listdict, prng=None, mapping=None):
    prng = np.random.default_rng(prng)
    if mapping is None:
        vars_set = set(n for g in generators_listdict for n in g[0].keys())
        mapping = {n: n for n in vars_set}
    for generator, len_orbit in generators_listdict:
        # Generator is a dictionary, with some given orbit length
        # e.g. {2: 4, 4: 2}
        for _ in range(prng.integers(len_orbit)):
            mapping.update({k: mapping[v] for k, v in generator.items()})

    return mapping


def prune_by_vacancies(proposal, nodeset):
    """Remove mappings for absent nodes, and delete invalidated maps."""
    # Removing as a function of edge defects would also be useful.

    nodeset = nodeset.intersection(set(proposal[0].keys()))
    if not nodeset:
        return None
    new = {n: proposal[0][n] for n in nodeset}
    if nodeset == set(new.values()):
        return (new, proposal[1])
    else:
        return None


class AutomorphismComposite(ComposedSampler):
    """Composite for applying automorphic preprocessing.

    The ordering of variables is permuted subject to a given pattern of
    generators or an inherited sampler structure.
    This can be useful to mitigate for noise, control errors or other
    symmetry-breaking features of the child sampler.

    .. note::
        If you are configuring an anneal schedule, be mindful that this
        composite does not recognize the ``initial_state`` parameter
        used by dimod's :class:`~dwave.system.samplers.DWaveSampler` for
        reverse annealing (composites do not generally process all keywords
        of child samplers) and does not permute any of the configured initial
        states.

    Args:
        sampler: A `dimod` sampler object.

        seed: As passed to :func:`numpy.random.default_rng`.

        generators_listdict: A set of permutations compatible with the child
            strcture. A list where each element is a tuple of generator (dict)
            and integer orbit length. This allows for uniform sampling of some
            graphs. If generators_listdict is None (by default) a schreir_context
            is used.

        schreir_context: A dwave.experimental.SchreirContext object. This allows
            fair sampling of any graph. The SchreirContext for an arbitrary
            graph can be created using dwave.experimental. If a schreir_context
            is None, and generators_listdict is None, then the schreir_context
            compatible with the child sampler structure is created by default.
            If both a generators_listdict, and a schreir_context are provided,
            the generators_listdict is ignored.

    Examples:
        This example composes a dimod ExactSolver sampler with automorphims then
        uses it to sample an Ising problem.

        >>> from dimod import ExactSolver
        >>> from dwave.preprocessing.composites import AutomorphismComposite
        >>> base_sampler = ExactSolver()
        >>> generators_listdict = [({'a': 'b', 'b':'a'}, 2)]
        >>> composed_sampler = AutomorphismComposite(base_sampler, generators_listdict)
        ... # Sample an Ising problem
        >>> response = composed_sampler.sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
        >>> response.first.sample
        {'a': -1, 'b': -1}

    References
    ----------
    .. [#TODO]
    """

    _children: list[dimod.core.Sampler]
    _parameters: dict[str, Sequence[str]]
    _properties: dict[str, Any]

    def __init__(
        self,
        child: dimod.core.Sampler,
        *,
        seed=None,
        generators_listdict: list[tuple[dict, int]] = None,
        generators_u_vector: list[list[np.ndarray[np.intp]]] = None,
        G: nx.Graph = None,
        idx_to_node: dict = None,
    ):
        self._child = child
        self.rng = np.random.default_rng(seed)
        self.generators_listdict = generators_listdict
        self.generators_u_vector = generators_u_vector
        self.idx_to_node = idx_to_node
        if (
            generators_u_vector is None
            and generators_listdict is None
            and G is not None
        ):
            self.idx_to_node = {idx: n for idx, n in enumerate(G.nodes)}
            result = schreier_rep(
                nx.relabel_nodes(G, {n: idx for idx, n in self.idx_to_node.items()}),
                num_samples=G.number_of_nodes(),
            )
            self.generators_u_vector = result.u_vector

    @property
    def children(self) -> list[dimod.core.Sampler]:
        try:
            return self._children
        except AttributeError:
            pass

        self._children = children = [self._child]
        return children

    @property
    def parameters(self) -> dict[str, Sequence[str]]:
        try:
            return self._parameters
        except AttributeError:
            pass

        self._parameters = parameters = dict(automorphism_variables=tuple())
        parameters.update(self._child.parameters)
        return parameters

    @property
    def properties(self) -> dict[str, Any]:
        try:
            return self._properties
        except AttributeError:
            pass

        self._properties = dict(child_properties=self._child.properties)
        return self._properties

    class _SampleSets:
        def __init__(self, samplesets: list[dimod.SampleSet]):
            self.samplesets = samplesets

        def done(self) -> bool:
            return all(ss.done() for ss in self.samplesets)

    @staticmethod
    def _reorder_variables(
        sampleset: dimod.SampleSet, order: dimod.variables.Variables
    ) -> dimod.SampleSet:
        """Return a sampleset with the given variable order."""
        if sampleset.variables == order:
            return sampleset

        # .index(...) is O(1) for dimod's Variables objects so this isn't too bad
        sampleset_order = sampleset.variables
        reorder = [sampleset_order.index(v) for v in order]

        return dimod.SampleSet.from_samples(
            (sampleset.record.sample[:, reorder], order),
            sort_labels=False,
            vartype=sampleset.vartype,
            info=sampleset.info,
            **sampleset.data_vectors,
        )

    @dimod.decorators.nonblocking_sample_method
    def sample(
        self,
        bqm: dimod.BinaryQuadraticModel,
        *,
        mappings: Optional[list[dict]] = None,
        num_automorphisms: Optional[int] = None,
        **kwargs,
    ):
        """Sample from the binary quadratic model.

        Args:
            bqm: Binary quadratic model to be sampled from.

            mappings:
                A list of mappings in the form of dictionaries.
                Each dictionary defines a permutation over a
                subset of variables. If mappings is provided and
                length 0, then a sampleset is returned subject
                to no automorphism.

            num_automorphisms:
                When mappings is not given, specifies teh number of mappings to
                apply (create). If mappings is provided, it
                is inferred as :code:`len(mappings)`, otherwise it is defaulted
                to 1.
                A value of ``0`` will result in sampling of an unmapped problem.
                If mappings is None the mappings are generated randomly using the
                `generators_listdict` class variable.

        Returns:
            A sample set. Note that for a sampler that returns ``num_reads`` samples,
            the sample set will contain ``num_reads*num_automorphisms`` samples.

        Examples:
            This example runs 100 automorphisms applied to one variable of a QUBO problem.

            >>> from dimod import ExactSolver
            >>> from dwave.preprocessing.composites import EmbeddingComposite
            >>> base_sampler = ExactSolver()
            >>> composed_sampler = EmbeddingComposite(base_sampler)
            ...
            >>> Q = {('a', 'a'): -1, ('b', 'b'): -1, ('a', 'b'): 2}
            >>> response = composed_sampler.sample_qubo(Q,
            ...               num_automorphisms=10)
            >>> len(response)
            40
        """
        sampler = self._child

        if num_automorphisms is None:
            if mappings is not None:
                num_automorphisms = len(mappings)
            else:
                num_automorphisms = 1

        # No SRTs, so just pass the problem through
        if not num_automorphisms or not bqm.num_variables:
            sampleset = sampler.sample(bqm, **kwargs)
            # yield twice because we're using the @nonblocking_sample_method
            yield sampleset  # this one signals done()-ness
            yield sampleset  # this is the one actually used by the user
            return

        # Check or generate mappings.
        if mappings is not None:
            # Given permutation
            if len(mappings) != num_automorphisms:
                raise ValueError(
                    "len(mappings) should match num_automorphisms when not None"
                )
        elif self.generators_listdict is not None:
            # Generator compatible (uniform random) permutation on all variables
            mapping = {v: v for v in bqm.variables}
            mappings = [
                sample_automorphisms_listdict(
                    self.generators_listdict, prng=self.rng, mapping=mapping
                )
                for _ in range(num_automorphisms)
            ]
        elif self.generators_u_vector is not None:
            arrays = sample_automorphisms_u_vectors(
                self.generators_u_vector, num_samples=num_automorphisms
            )
            mappings = [
                {self.idx_to_node[k]: self.idx_to_node[v] for k, v in enumerate(array)}
                for array in arrays
            ]
        else:
            # Random permutation (no generator constraint) on all variables
            var_list = list(bqm.variables)
            mappings = []
            for n in range(num_automorphisms):
                self.rng.shuffle(var_list)
                mappings.append({v1: v2 for v1, v2 in zip(bqm.variables, var_list)})

        samplesets: list[dimod.SampleSet] = []
        for mapping in mappings:
            _bqm = bqm.copy()  # Costly, but assumed not to be a bottleneck
            _bqm.relabel_variables(mapping)
            samplesets.append(sampler.sample(_bqm, **kwargs))
        # Yield a view of the samplesets that reports done()-ness
        yield self._SampleSets(samplesets)

        # Relabel variables on samplesets
        for mapping, ss in zip(mappings, samplesets):
            ss.relabel_variables({v: k for k, v in mapping.items()})

        # Reorder the variables of all the returned samplesets to match our
        # original BQM
        samplesets = [self._reorder_variables(ss, bqm.variables) for ss in samplesets]

        if num_automorphisms == 1:
            # If one sampleset, return full information
            # (info returned in full)
            yield samplesets[0]
        else:
            # finally combine all samplesets together
            yield dimod.concatenate(samplesets)


if __name__ == "__main__":
    from dwave_networkx import chimera_graph, zephyr_graph, pegasus_graph
    from networkx import relabel_nodes
    from dimod import ExactSolver
    from itertools import product

    # QUITE THOROUGH: MOVE THIS TO TESTS

    base_sampler = ExactSolver()

    G = nx.from_edgelist([("a", "b")])
    composed_sampler = AutomorphismComposite(base_sampler, G=G)
    print(composed_sampler.generators_u_vector)
    print(
        sample_automorphisms_u_vectors(
            composed_sampler.generators_u_vector, 10, rng=None
        )
    )
    response = composed_sampler.sample_ising({"a": -0.5, "b": 1.0}, {("a", "b"): -1})
    print(response.first.sample)

    for generators_listdict in [None, [({"a": "b", "b": "a"}, 2)]]:
        composed_sampler = AutomorphismComposite(
            base_sampler, generators_listdict=generators_listdict
        )
        # Sample an Ising problem
        response = composed_sampler.sample_ising(
            {"a": -0.5, "b": 1.0}, {("a", "b"): -1}
        )
        print(response.first.sample)

    # Chimera_cell:
    from dwave.system.testing import MockDWaveSampler
    from dwave_networkx import (
        chimera_coordinates,
        zephyr_coordinates,
        pegasus_coordinates,
        draw_chimera,
    )
    import matplotlib.pyplot as plt

    # Test chimera generators:
    topology_type = "chimera"
    # topology_type = "zephyr"
    # topology_type = "pegasus"
    if topology_type == "chimera":
        topology_shape = [3, 2, 3]
        make_graph = chimera_graph
        graph_generators = chimera_generators
        coord_transform = chimera_coordinates(*topology_shape).chimera_to_linear
        shapes = product([1, 3], [1, 3])
    elif topology_type == "zephyr":
        topology_shape = [3, 2]
        make_graph = zephyr_graph
        graph_generators = zephyr_generators
        coord_transform = zephyr_coordinates(*topology_shape).zephyr_to_linear
        shapes = product([1, 3], [1, 3])
    elif topology_type == "pegasus":
        topology_shape = [2]
        make_graph = pegasus_graph
        graph_generators = pegasus_generators
        coord_transform = pegasus_coordinates(*topology_shape).pegasus_to_linear
        shapes = [[2], [3]]
    else:
        raise ValueError("Unknown topology")
    base_sampler = MockDWaveSampler(
        topology_type=topology_type, topology_shape=topology_shape
    )
    G = base_sampler.to_networkx_graph()
    # draw_chimera(G, with_labels=True)
    # G2 = relabel_nodes(G, {n: chimera_coordinates(*topology_shape).linear_to_chimera(n) for n in G.nodes()})
    # draw_chimera(G2, with_labels=True)
    # plt.show()

    Gedges = set(tuple(sorted(e)) for e in G.edges())
    Gnodes = set(G.nodes())
    generators = graph_generators(*topology_shape)
    generators = [
        ({coord_transform(k): coord_transform(v) for k, v in g[0].items()}, g[1])
        for g in generators
    ]
    pruned_generators = []
    for g in generators:
        new = prune_by_vacancies(proposal=g, nodeset=Gnodes)
        if new:
            pruned_generators.append(new)
            assert set(new[0].keys()).issubset(Gnodes)
    composed_sampler = AutomorphismComposite(
        base_sampler, generators_listdict=pruned_generators
    )

    composed_sampler2 = AutomorphismComposite(base_sampler, G=G)
    print("u_vector", composed_sampler2.generators_u_vector)

    bqm = dimod.BinaryQuadraticModel("SPIN").from_ising({}, {e: -1 for e in G.edges})
    for cs in [composed_sampler, composed_sampler2]:
        response = composed_sampler.sample(bqm, num_automorphisms=2)
        print(response.first.sample)

    for shape in shapes:
        if len(shape) == 2:
            m, t = shape
            graph_params = {"m": shape[0], "t": shape[1]}
        else:
            m = shape
            graph_params = {"m": shape[0]}
        G = make_graph(**graph_params, coordinates=True)
        Gedges = set(tuple(sorted(e)) for e in G.edges())
        Gnodes = set(G.nodes())
        generators = graph_generators(**graph_params)
        pruned_generators = []
        for g in generators:
            new = prune_by_vacancies(proposal=g, nodeset=Gnodes)
            if new:
                pruned_generators.append(new)
                assert set(new[0].keys()).issubset(Gnodes)

        for g, ol in pruned_generators:
            if set(g.keys()) != set(g.values()):
                print(g.keys())
                print(g.values())
            assert set(g.keys()) == set(g.values())
            Gn = relabel_nodes(G, g)
            Gnedges = set(tuple(sorted(e)) for e in G.edges())
            if set(Gn.nodes()) != Gnodes:
                print(g, Gnodes)
                print("Gnodes Gn.nodes difference", Gnodes.difference(set(Gn.nodes())))
                print("Gnodes, Gnedges difference", set(Gn.nodes()).difference(Gnedges))
            assert set(Gn.nodes()) == Gnodes
            if Gnedges != Gedges:
                print(sorted(set(tuple(sorted(e)) for e in Gedges)))
                print(sorted(set(G.edges())))
                print(Gnedges.difference(Gedges))
                print(Gedges.difference(Gnedges))
            assert Gnedges == Gedges
            # assert list(Gn.edges()) != list(G.edges())

        random_perm = sample_automorphisms_listdict(pruned_generators)
        assert set(random_perm.keys()) == Gnodes
        assert set(random_perm.values()) == Gnodes
        Gn = relabel_nodes(G, random_perm)
        if set(Gn.nodes()) != Gnodes:
            print("Problem")
            print(Gnodes.difference(set(Gn.nodes())))
            print(set(Gn.nodes()).difference(Gnedges))

        assert set(Gn.nodes()) == Gnodes
        Gnedges = set(tuple(sorted(e)) for e in G.edges())
        assert Gnedges == Gedges
