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

import typing

import dimod
import numpy as np

from dimod import Vartype, ComposedSampler

__all__ = ["AutomorphismComposite", "chimera_generators", "zephyr_generators", "generator_shuffle"]


def chimera_generators(m, n=None, t=4):
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


def flipZephyr(u, w, k, j, z, orient, m):
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


def zephyr_generators(m, t=4):
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


def generator_shuffle(generators, prng=None, vars_map=None):
    prng = np.random.default_rng(prng)
    if vars_map is None:
        vars_set = set(n for g in generators for n in g[0].keys())
        vars_map = {n: n for n in vars_set}
    for generator, len_orbit in generators:
        # Generator is a dictionary, with some given orbit length
        # e.g. {2: 4, 4: 2}
        for _ in range(prng.integers(len_orbit)):
            vars_map.update({k: vars_map[v] for k, v in generator.items()})

    return vars_map


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

        generators: A set of permutations compatible with the child structure.

    Examples:
        This example composes a dimod ExactSolver sampler with automorphims then
        uses it to sample an Ising problem.

        >>> from dimod import ExactSolver
        >>> from dwave.preprocessing.composites import AutomorphismComposite
        >>> base_sampler = ExactSolver()
        >>> composed_sampler = AutomorphismComposite(base_sampler)
        ... # Sample an Ising problem
        >>> response = composed_sampler.sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
        >>> response.first.sample
        {'a': -1, 'b': -1}

    References
    ----------
    .. [#km] Andrew D. King and Catherine C. McGeoch. Algorithm engineering
        for a quantum annealing platform. https://arxiv.org/abs/1410.2628,
        2014.

    """

    _children: typing.List[dimod.core.Sampler]
    _parameters: typing.Dict[str, typing.Sequence[str]]
    _properties: typing.Dict[str, typing.Any]

    def __init__(self, child: dimod.core.Sampler, *, seed=None, generators=None):
        self._child = child
        self.rng = np.random.default_rng(seed)
        self.generators = generators

    @property
    def children(self) -> typing.List[dimod.core.Sampler]:
        try:
            return self._children
        except AttributeError:
            pass

        self._children = children = [self._child]
        return children

    @property
    def parameters(self) -> typing.Dict[str, typing.Sequence[str]]:
        try:
            return self._parameters
        except AttributeError:
            pass

        self._parameters = parameters = dict(automorphism_variables=tuple())
        parameters.update(self._child.parameters)
        return parameters

    @property
    def properties(self) -> typing.Dict[str, typing.Any]:
        try:
            return self._properties
        except AttributeError:
            pass

        self._properties = dict(child_properties=self._child.properties)
        return self._properties

    class _SampleSets:
        def __init__(self, samplesets: typing.List[dimod.SampleSet]):
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
        num_automorphisms: int = 1,
        **kwargs,
    ):
        """Sample from the binary quadratic model.

        Args:
            bqm: Binary quadratic model to be sampled from.

            num_automorphisms:
                Number of automorphisms.
                A value of ``0`` will not transform the problem.
                If you specify a nonzero value, each automorphism
                will result in an independent run of the child sampler.

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
            ...               num_automorphisms=100)
            >>> len(response)
            400
        """
        sampler = self._child

        # No SRTs, so just pass the problem through
        if not num_automorphisms or not bqm.num_variables:
            sampleset = sampler.sample(bqm, **kwargs)
            # yield twice because we're using the @nonblocking_sample_method
            yield sampleset  # this one signals done()-ness
            yield sampleset  # this is the one actually used by the user
            return

        # we'll be modifying the BQM, so make a copy
        bqm = bqm.copy()

        # We maintain the Leap behavior that num_automorphisms == 1
        # corresponds to a single problem with randomly flipped variables.

        # Submit the problems
        samplesets: typing.List[dimod.SampleSet] = []
        if self.generators is not None:
            vars_map = {i: i for i in bqm.variables}
            for i in range(num_automorphisms):
                relabeling = generator_shuffle(
                    self.generators, prng=self.rng, vars_map=vars_map
                )
                bqm.relabel_variables(relabeling)
                samplesets.append(sampler.sample(bqm, **kwargs))
        else:
            vars_copy = list(bqm.variables)
            for i in range(num_automorphisms):
                self.rng.shuffle(vars_copy)
                relabeling = {i: j for i, j in zip(bqm.variables, vars_copy)}
                bqm.relabel_variables(relabeling)
                samplesets.append(sampler.sample(bqm, **kwargs))

        # Yield a view of the samplesets that reports done()-ness
        yield self._SampleSets(samplesets)

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
    from dwave_networkx import chimera_graph, zephyr_graph
    from networkx import relabel_nodes
    from dimod import ExactSolver
    from itertools import product

    # from dwave.preprocessing.composites import AutomorphismComposite
    base_sampler = ExactSolver()
    generators = [({"a": "b", "b": "a"}, 2)]
    composed_sampler = AutomorphismComposite(base_sampler, generators=generators)
    # Sample an Ising problem
    response = composed_sampler.sample_ising({"a": -0.5, "b": 1.0}, {("a", "b"): -1})
    print(response.first.sample)

    # Chimera_cell:
    from dwave.system.testing import MockDWaveSampler
    from dwave_networkx import chimera_coordinates, zephyr_coordinates, draw_chimera
    import matplotlib.pyplot as plt

    # Test chimera generators:
    if False:
        topology_type = "chimera"
        topology_shape = [3, 2, 3]
        make_graph = chimera_graph
        graph_generators = chimera_generators
        coord_transform = chimera_coordinates(*topology_shape).chimera_to_linear
    else:
        topology_type = "zephyr"
        topology_shape = [3, 2]
        make_graph = zephyr_graph
        graph_generators = zephyr_generators
        coord_transform = zephyr_coordinates(*topology_shape).zephyr_to_linear
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

    composed_sampler = AutomorphismComposite(base_sampler, generators=generators)
    bqm = dimod.BinaryQuadraticModel("SPIN").from_ising({}, {e: -1 for e in G.edges})
    response = composed_sampler.sample(bqm, num_automorphisms=2)
    print(response.first.sample)

    for t, m in product([1, 3], [1, 3]):
        print(m, t)
        graph_params = {"m": m, "t": t, "coordinates": True}
        G = make_graph(**graph_params)
        Gedges = set(tuple(sorted(e)) for e in G.edges())
        Gnodes = set(G.nodes())
        generators = graph_generators(m=m, t=t)
        for g, ol in generators:
            if set(g.keys()) != set(g.values()):
                print(g.keys())
                print(g.values())
            assert set(g.keys()) == set(g.values())
            Gn = relabel_nodes(G, g)
            if set(Gn.nodes()) != Gnodes:
                print(Gnodes.difference(set(Gn.nodes())))
                print(set(Gn.nodes()).difference(Gnedges))
            assert set(Gn.nodes()) == Gnodes
            Gnedges = set(tuple(sorted(e)) for e in G.edges())
            if Gnedges != Gedges:
                print(sorted(set(tuple(sorted(e)) for e in Gedges)))
                print(sorted(set(G.edges())))
                print(Gnedges.difference(Gedges))
                print(Gedges.difference(Gnedges))
            assert Gnedges == Gedges
            # assert list(Gn.edges()) != list(G.edges())
        random_perm = generator_shuffle(generators)
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
