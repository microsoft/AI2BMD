# Fragmentation

In AI2BMD, the input protein is fragmented into a number of dipeptides and
ACE-NMEs, which are fed to the ViSNet model. In general, each dipeptide
fragment consists of the backbone of the previous amino acid residue, all atoms
of the current amino acid residue, and the backbone of the next amino acid
residue, while each ACE-NME fragment consists of the acetyl group of an amino
acid and the N-methyl amide group of the previous amino acid.

## Input

The fragmentation process takes a protein as input, in the form of a PDB file.
This file contains The columns are, from left to right, record type, atom
index, atom name, residue type, residue index, cartesian coordinates,
occupancy, temperature factor, and atom type.

```
ATOM      1  H1  ACE     1      10.845   8.614   5.964  1.00  0.00           H
ATOM      2  CH3 ACE     1      10.143   9.373   5.620  1.00  0.00           C
ATOM      3  H2  ACE     1       9.425   9.446   6.437  1.00  0.00           H
ATOM      4  H3  ACE     1       9.643   9.085   4.695  1.00  0.00           H
ATOM      5  C   ACE     1      10.805  10.740   5.408  1.00  0.00           C
ATOM      6  O   ACE     1      10.682  11.417   4.442  1.00  0.00           O
ATOM      7  N   TYR     2      11.363  11.214   6.507  1.00  0.00           N
ATOM      8  H   TYR     2      11.345  10.675   7.361  1.00  0.00           H
ATOM      9  CA  TYR     2      11.963  12.513   6.704  1.00  0.00           C
ATOM     10  HA  TYR     2      12.444  12.615   5.731  1.00  0.00           H
ATOM     11  CB  TYR     2      10.909  13.631   6.882  1.00  0.00           C
ATOM     12  HB2 TYR     2      10.230  13.593   6.030  1.00  0.00           H
ATOM     13  HB3 TYR     2      10.264  13.332   7.708  1.00  0.00           H
ATOM     14  CG  TYR     2      11.383  15.023   7.089  1.00  0.00           C
ATOM     15  CD1 TYR     2      11.729  15.440   8.404  1.00  0.00           C
ATOM     16  HD1 TYR     2      11.527  14.718   9.182  1.00  0.00           H
ATOM     17  CE1 TYR     2      12.033  16.805   8.578  1.00  0.00           C
ATOM     18  HE1 TYR     2      12.380  17.146   9.543  1.00  0.00           H
ATOM     19  CZ  TYR     2      12.136  17.704   7.498  1.00  0.00           C
ATOM     20  OH  TYR     2      12.556  19.002   7.619  1.00  0.00           O
ATOM     21  HH  TYR     2      12.437  19.327   8.514  1.00  0.00           H
ATOM     22  CE2 TYR     2      11.872  17.214   6.251  1.00  0.00           C
ATOM     23  HE2 TYR     2      12.034  17.842   5.387  1.00  0.00           H
ATOM     24  CD2 TYR     2      11.440  15.923   6.033  1.00  0.00           C
ATOM     25  HD2 TYR     2      11.349  15.559   5.021  1.00  0.00           H
ATOM     26  C   TYR     2      13.070  12.496   7.702  1.00  0.00           C
ATOM     27  O   TYR     2      12.959  11.951   8.788  1.00  0.00           O
...
```

## Extracting indices for each fragment

The first step is to extract the indices of the atoms that belong to each
dipeptide and ACE-NME fragment (implementation in
`src/Fragmentation/basefrag.py:get_fragments_index`). We loop through the list
of atoms in order, and insert the atom into the list of dipeptides and ACE-NMEs
as appropriate. For example, the `CA` and `HA` atoms belong in the dipeptide
and both ACE/NME fragments, while the `C` and `O` atoms belong only to the
dipeptide and ACE (acetyl group) part of the fragment. The side-chain of each
amino acid is present in only one of the dipeptides. Care is taken to calculate
the correct dipeptide/ACE/NME index that each atom belongs to, and that edge
cases (first/last dipeptides/ACE-NMEs) are handled properly.

The following diagram illustrates the division of atoms.

```
     H   O   H   R   O   H   R   O   H   R   O   H   R   O
     |   "   |   |   "   |   |   "   |   |   "   |   |   "
 H - C - C - N - C - C - N - C - C - N - C - C - N - C - C - ...
     |           |           |           |           |
     H           H           H           H           H
|_________| |_________| |_________| |_________| |_________|
    ACE         TYR         TYR         ASP         PRO

|------- 1st dipeptide -------|
                |----- 2nd dipeptide -----|
                            |----- 3rd dipeptide -----|
                                        |----- 4th dipeptide -----|

                |-------------|
                      1st   |-------------|
                    ACE-NME       2nd   |-------------|
                                ACE-NME       3rd
                                            ACE-NME
```

## Capping dipeptides with hydrogens

The dipeptide and ACE-NME fragments are extracted directly from the full amino
acid chain, and thus have dangling bonds at the edges, which is inconsistent
with their original conditions. Hence, additional hydrogen atoms are added
where necessary to avoid artifical interactions resulting from the incomplete
covalent bonds.

As a first approximation, the hydrogens are added along the same direction as
the original bond, with the bond length modified to be the sum of the average
radii of the two atoms. The coordinates of the added hydrogen atoms are later
optimised to relax the system. Since the ACE-NME fragments overlap entirely
with the dipeptide fragments, the optimisation is performed only for the
dipeptide fragments.

```
      R   O   H   R   O   H   R   O   H   R   O   H   R   O
      |   "   |   |   "   |   |   "   |   |   "   |   |   "
... - C - C - N - C - C - N - C - C - N - C - C - N - C - C - ...
      |           |           |           |           |
      H           H           H           H           H
                 |------- dipeptide -------|

                 *H   O   H   R   O   H   H*
                  |   "   |   |   "   |   |
             *H - C - C - N - C - C - N - C - H*
                  |           |           |
                  H           H           H
```

In the example above, the dipeptide fragment has four broken bonds: 2 each on
the alpha-carbons on the first and last residues that result from the removal
of the side-chains and the neighbouring aminde and carboxyl groups.

The number of added hydrogen atoms depends on the specific amino acid residues
that make up the fragments. For example, glycine (GLY) already has a single
hydrogen atom as its side-chain, and thus does not require a hydrogen atom to
be added.

```
CA C O HA N CA C O H HA CB CG OD1 OD2 HB2 HB3 N CA HA | [H*] [H*] [H*] [H*] [H*]

 CA C [H*] [H*] O HA N CA C O H HA CB CG OD1 OD2 HB2 HB3 N [H*] CA HA [H*] [H*]
```

## Reordering atoms

The atoms in the dipeptide fragments have to be reordered to match the format
that AMBER expects. This reordering of indices is implemented using a
precomputed lookup table (`src/utils/seq_dict.pkl`) based on the three residue
types that make up the dipeptide fragment.

```
 CA C [H*] [H*] O HA N CA C O H HA CB CG OD1 OD2 HB2 HB3 N [H*] CA HA [H*] [H*]

 HA CA [H*] [H*] C O N H CA HA CB HB2 HB3 CG OD1 OD2 C O N [H*] CA HA [H*] [H*]
```

In our implementation, the insertion of hydrogen atoms and reordering of the
dipeptide is performed in a single step.

```
CA C O HA N CA C O H HA CB CG OD1 OD2 HB2 HB3 N CA HA | [H*] [H*] [H*] [H*] [H*]

 HA CA [H*] [H*] C O N H CA HA CB HB2 HB3 CG OD1 OD2 C O N [H*] CA HA [H*] [H*]
```

The ACE-NME fragments can then be obtained by extracting the first and last six
atoms of the appropriate dipeptides, e.g. the 1st ACE-NME fragment consists of
the first six atoms of the 2nd dipeptide and the last six atoms of the 1st
dipeptide.
