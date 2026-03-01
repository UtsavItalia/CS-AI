import csv
import itertools
import sys

PROBS = {
    # Unconditional probabilities for having gene
    "gene": {2: 0.01, 1: 0.03, 0: 0.96},
    "trait": {
        # Probability of trait given two copies of gene
        2: {True: 0.65, False: 0.35},
        # Probability of trait given one copy of gene
        1: {True: 0.56, False: 0.44},
        # Probability of trait given no gene
        0: {True: 0.01, False: 0.99},
    },
    # Mutation probability
    "mutation": 0.01,
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {"gene": {2: 0, 1: 0, 0: 0}, "trait": {True: 0, False: 0}}
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):
        # Check if current set of people violates known information
        fails_evidence = any(
            (
                people[person]["trait"] is not None
                and people[person]["trait"] != (person in have_trait)
            )
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (
                    True
                    if row["trait"] == "1"
                    else False
                    if row["trait"] == "0"
                    else None
                ),
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s)
        for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    jp = 1
    zero_gene = set(people) - one_gene.union(two_genes)
    inherit_chances = {0: PROBS["mutation"], 1: 0.5, 2: 1 - PROBS["mutation"]}
    for name in zero_gene:
        if people[name]["mother"] is None and people[name]["father"] is None:
            jp *= PROBS["gene"][0]
        else:
            motherName = people[name]["mother"]
            fatherName = people[name]["father"]
            mother_passes = inherit_chances[0 if motherName in zero_gene else 1 if motherName in one_gene else 2]
            father_passes = inherit_chances[0 if fatherName in zero_gene else 1 if fatherName in one_gene else 2]
            #mother or father doesnt passes
            jp *= (1 - mother_passes) * (1 - father_passes)

        jp *= PROBS["trait"][0][name in have_trait]

    for name in one_gene :
        if people[name]["mother"] is None and people[name]["father"] is None:
            jp *= PROBS["gene"][1]
        else:
            motherName = people[name]["mother"]
            fatherName = people[name]["father"]
            mother_passes = inherit_chances[0 if motherName in zero_gene else 1 if motherName in one_gene else 2]
            father_passes = inherit_chances[0 if fatherName in zero_gene else 1 if fatherName in one_gene else 2]
            # mother passes but father doesnt, or father passes and mother doesnt
            jp *= mother_passes * (1 - father_passes) + father_passes * (1 - mother_passes)

        jp *= PROBS["trait"][1][name in have_trait]

    for name in two_genes:
        if people[name]["mother"] is None and people[name]["father"] is None:
            jp *= PROBS["gene"][2]
        else:
            motherName = people[name]["mother"]
            fatherName = people[name]["father"]
            mother_passes = inherit_chances[0 if motherName in zero_gene else 1 if motherName in one_gene else 2]
            father_passes = inherit_chances[0 if fatherName in zero_gene else 1 if fatherName in one_gene else 2]
            #both parents must pass
            jp *= mother_passes * father_passes

        jp *= PROBS["trait"][2][name in have_trait]
    return jp


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    zero_gene = set(probabilities) - one_gene.union(two_genes)
    for name in one_gene:
        probabilities[name]["gene"][1] += p
        if name in have_trait:
            probabilities[name]["trait"][True] += p
        else:
            probabilities[name]["trait"][False] += p

    for name in two_genes:
        probabilities[name]["gene"][2] += p
        if name in have_trait:
            probabilities[name]["trait"][True] += p
        else:
            probabilities[name]["trait"][False] += p

    for name in zero_gene:
        probabilities[name]["gene"][0] += p
        if name in have_trait:
            probabilities[name]["trait"][True] += p
        else:
            probabilities[name]["trait"][False] += p



def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for name in probabilities:
        gene_sum = 0
        for gene in  probabilities[name]["gene"]:
            gene_sum += probabilities[name]["gene"][gene]
        gene_multiplier = 1 / gene_sum
        for gene in probabilities[name]["gene"]:
            probabilities[name]["gene"][gene] *= gene_multiplier
        trait_sum = 0

        for trait in  probabilities[name]["trait"]:
            trait_sum += probabilities[name]["trait"][trait]
        trait_multiplier = 1 / trait_sum
        for trait in probabilities[name]["trait"]:
            probabilities[name]["trait"][trait] *= trait_multiplier

if __name__ == "__main__":
    main()
