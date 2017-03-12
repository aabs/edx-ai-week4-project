# Using a Prairie Fire Algorithm to Gauge the Utility of a 2048 Board Position

There are a couple of intuitions about the quality of a board based on
contiguity of like tiles:

1. The ratio of tiles to contiguous groups is a measure of how fractured
    a board is.
2. contiguous regions of similar tiles are better than dissimilar ones.
3. The worst scenario is a highly factured board with dissimilar tiles
    adjacent to one another.
4. whilst having like tiles on the same rank and file with only zeros
    between is better than nothing, it costs a move to merge them, so
    they are not really that useful as a way to promote net negative
    growth

The PFA can be used to provide a utility measure by:

1. examining how fractured each tile type is
2. examining how many tiles of a type are together
3. providing info about how much space is available
4. providing a means to penalise scattered low number tiles that were
    likely introduced as a result of poorly chosen moves
5. being combined with monotonicity measures to ensure that not only
    are the tiles clustered in the correct place on the board, but
    can be merged too
6. subsuming measures like roughness/smoothness, max_tile, snake (since
    a well-formed snake would register as a single cluster)
7. providing a before/after comparison of a board to demonstrate that a
    chosen move resulted in a fractured board, or broke up some clusters

How can the utility score be calculated?

1. rewards for clusters, proportional to cluster size
2. penalties for fragmentation, proportional to number of fragments
3. rewards for larger tiles
4. rewards for keeping lower tiles in lower quantities
5. penalties for fragmenting the zeros (since that must mean space is
    limited and a cliff may have just been created, which could lead to
    a checkerboard pattern which is ultimately often fatal)
6. rewards for the size and shape of the clusters
7. rewards for what neighbours the cluster has

Some useful info that could be used to augment the results of the
algorithm:

1. what values were adjacent to a cluster (that might then be merged
    into)
2. the spatial dimensions of the cluster? i.e. round, thin, angled etc
3. how many turns a thin cluster takes.
4. whether the values on either side of the clusters form a monotonic
    sequence

The algorithm could capture this kind of info in the process of lighting
a fire - if a number found was not part of the cluster, then record what
it was and what orientation it had to the custer. The resultant info
would be only partially useful, but it could give a taste of the env
around a cluster that might be useful as an indicator of whether
merging the cluster in a certain direction was worthwhile

Does the way the results are used change through the course of the game?

# Prairie Fire Algorithm

Objectives:
- to gather data about:
    - the number of groups of tiles
    - how fractured each kind of tile is.
    - whether tile groups are adjacent to other tiles that they might merge with
- to work in concert with algorithms to:
    - detect features that are known to be bad for ongoing game play
    - detect monotonicity
    - detect cliffs

# Algorithm:
```python
def prairie_fire(g):
    def set_fire_to(B, x, y, t, code):
        # if off edge
        if x < 0 or x > 3 or y < 0 or y > 3:
            return False
        # if done already
        if B[x, y] == code:
            return False
        # if no match
        if B[x, y] != t:
            return False

        B[x, y] = code
        set_fire_to(B, x - 1, y, t, code)
        set_fire_to(B, x + 1, y, t, code)
        set_fire_to(B, x, y - 1, t, code)
        set_fire_to(B, x, y + 1, t, code)
        return True

    B = g.clone()
    result = dict()
    tiles = [2 ** l if l > 0 else 0 for l in range(0, 17)]
    tiles.insert(0, 0)  # add zero at the beginning, so we can also see how much space we have
    for t in tiles:
        result[t] = []
    for t in tiles:
        code = -1
        for x in range(0, 4):
            for y in range(0, 4):
                lit = set_fire_to(B, x, y, t, code)
                if lit:
                    code -= 1
        # now gather the stats
        for c in range(-1, code - 1, -1):
            count = 0
            for x in range(0, 4):
                for y in range(0, 4):
                    if B[x,y] == c:
                        count += 1
                        B[x,y] = 0
            if count > 0:
                result[t].append(count)
    return result
```