"""
Hierarchical Feature Engineering (HFE) Python3 implementation.
  - See paper: "Taxonomy-Aware Feature Engineering for Microbiome Classification".
  - This is largely the Oudah (2018) implementation, modified with new information gain functions to avoid WEKA.
  - Comments have been added to try to match blocks with the paper's stages.
  - The original code ferried data using text files, and that behavior has been retained.

"""
from __future__ import division
import os
import sys
import csv
import time
import warnings
import argparse
import math
import numpy as np
import pandas as pd
from collections import OrderedDict
from enum import Enum
from scipy.stats import pearsonr
from scipy.stats import ConstantInputWarning
from sklearn.feature_selection import mutual_info_classif
from datetime import timedelta

# Legacy imports. Use the enumeration "IGMethod.WEKA" to revert to original implementation.
from weka.core import jvm
from weka.core.converters import Loader
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection


# ----------------------------------------------------- Classes -------------------------------------------------------
#
#
class IGMethod(Enum):
    """
    Enumeration used in MATCH conditionals for computing the Information Gain using a given function.
    """
    WEKA = "weka"
    SCIKIT = "scikit"
    CUSTOM = "custom"

# ---------------------------------------------------- Constants ------------------------------------------------------
#
_DECIMAL_ROUND = 6

# ---------------------------------------------------- Functions ------------------------------------------------------
#
#
def get_print_time():
    """
    Helper function for returning a formatted date-time string for stdout messages.
    :return: The current time in the format: %d-%b-%Y %H:%M:%S
    """
    return time.strftime('%d-%b-%Y %H:%M:%S', time.localtime())


def into_tab(array, tab_dir, delim):
    """
    Writes a list to a tab-delimited file. From original implementation.
    :param array: The list to write.
    :param tab_dir: The directory to write the file into.
    :param delim: The delimiter to use.
    """
    ofilex = open(tab_dir, "wb")
    for i in range(len(array)):
        tmp = []
        for j in range(len(array[i])):
            tmp.append(array[i][j])
        out_str = str(tmp).replace("['", '').replace("']", '').replace(", ", delim).replace("'", '') + '\n'
        ofilex.write(out_str.encode())
    ofilex.close()


def get_IG(ofile_dir, loader):
    """
    Original Information Gain function.
    :param ofile_dir: Temp file with data for the features that will be tested.
    :param loader: A WEKA interface object that can read and load a CSV file.
    :return: Dictionary with information gain scores for each feature.
    """
    data = loader.load_file(ofile_dir)
    data.class_is_last()

    evaluator = ASEvaluation(classname="weka.attributeSelection.InfoGainAttributeEval")
    search = ASSearch(classname="weka.attributeSelection.Ranker", options=["-T", "-1.7976931348623157E308", "-N", "-1"])
    attsel = AttributeSelection()
    attsel.search(search)
    attsel.evaluator(evaluator)
    attsel.select_attributes(data)

    results = {}

    if attsel.number_attributes_selected < 2:
        flag = 0
        output = attsel.results_string
        for i in output.split('\n'):
            if flag != 0:
                if len(i.split(' ')) > 2:
                    t = []
                    for f in i.split(' '):
                        if f != '':
                            t.append(f)
                    r_tax = ''
                    for c in range(len(t)):
                        if c > 1:
                            r_tax = r_tax + t[c] + ' '
                    results.update({str(r_tax.strip()): float(t[0].strip())})
                else:
                    break
            if "Ranked attributes" in i:
                flag = 1
        mean_score = sum(results.values()) / len(results.values())
        os.system("rm -r " + ofile_dir)
    else:
        results = dict([(str(data.attribute(attr[0]).name), attr[1]) for attr in attsel.ranked_attributes])
        mean_score = attsel.ranked_attributes[:, 1].mean()

    return results, mean_score


def get_mutual_info(ofile_dir):
    """
    This function replaces the original code's function which calculated the Information Gain. This
    function implements the 'Expected Mutual Information', which is calculated between two variables and
    measures the reduction in uncertainty for one variable given a known value of the other variable.
    Note: Mutual Information and Information Gain are the same thing, although the context or usage of the measure
    often gives rise to the different names.

    The expected mutual information between two random variables X and Y is:
        I(X ; Y) = H(X) – H(X | Y)
        Where I(X ; Y) is the mutual information for X and Y, H(X) is the entropy for X and H(X | Y) is the conditional
        entropy for X given Y. The result has the units of bits.

    The entropy is calculated as H(X) = -sum(Pi*log2(Pi)), with Pi being the probability of the class i in the dataset.
    Entropy basically measures the degree of "impurity". The closest to 0 it is, the less impurity there is in
    your dataset. This function uses 'log2' whereas the old Weka-based code used 'loge' (natural log, ln).

    :param ofile_dir: Temporary file (from original design & implementation) that stores attributes and their values.
    :return: 'ig_results' is a dictionary of attributes and their values. 'ig_mean_score' is the mean of the computed
                scores in ''ig_results.
    """

    random_state = 42

    X = []
    y_targets = []

    with open(ofile_dir) as IG_FILE_POINTER:
        ig_file_reader = csv.reader(IG_FILE_POINTER, delimiter=',')

        file_header = next(ig_file_reader)
        file_header.pop()

        for line in ig_file_reader:  # 'line' is an array of the line's token fields. LABEL is last token.
            label = line.pop()  # last element is the class label. <STR>.
            y_targets.append(label)
            #   The rest of the line is an array of Strings which represent numeric values. They need to be
            #   converted from STR -> Float so that we can use them.
            X.append(list(map(float, line)))

    #   Call the SciKit function for computing mutual information.
    mi = mutual_info_classif(X=X,
                             y=y_targets,
                             discrete_features='auto',
                             random_state=random_state)

    K = _DECIMAL_ROUND  # How many K decimal values should we round off to?
    ig_results = dict(zip(file_header, mi.tolist()))
    ig_results_rounded = {key: round(ig_results[key], K) for key in ig_results}

    ig_mean_score = sum(ig_results_rounded.values()) / len(ig_results_rounded)

    return ig_results_rounded, ig_mean_score


def calc_entropy(column):
    """
    Calculate entropy given a pandas series, list, or numpy array.
    """
    # Compute the counts of each unique value in the column
    # counts = np.bincount(column)
    # counts = np.bincount(pd.factorize(l)[0])
    counts = np.bincount(pd.factorize(column)[0])

    # Divide by the total column length to get a probability
    probabilities = counts / len(column)

    # Initialize the entropy to 0
    entropy = 0
    # Loop through the probabilities, and add each one to the total entropy
    for prob in probabilities:
        if prob > 0:
            #   If no second argument, 'log()' will compute a natural logarithm.
            # entropy += prob * math.log(prob, 2)
            entropy += prob * math.log(prob)

    return -entropy


def calculate_information_gain(data, split_name, target_name):
    """
    Calculates the information gain given a data set, column to split on, and target.:
    :param data: The dataset.
    :param split_name: Name of the column we want to split on.
    :param target_name: Name of our target column.
    :return: The difference of the summation of the product of the probability and the subsets target column's
                entropy being subtracted from the original entropy
    """
    # Calculate the original entropy
    original_entropy = calc_entropy(data[target_name])

    to_subtract = 0

    if (data[split_name] == 0).all():
        to_subtract = 0
    else:
        #   The median is found for the column being split on. Any rows where the value of the variable is below
        #   the median will go to the left branch, and the rest of the rows will go to the right branch.
        #   To compute information gain, we’ll only have to compute entropies for two subsets.
        median = data[split_name].median()

        left_split = data[data[split_name] <= median]
        right_split = data[data[split_name] >= median]

        # Loop through the splits and calculate the subset entropies
        for subset in [left_split, right_split]:
            prob = (subset.shape[0] / data.shape[0])
            to_subtract += prob * calc_entropy(subset[target_name])

    # Return information gain
    return original_entropy - to_subtract


def information_gain(df_data):
    """
    Driver function for computing the Information Gain on a taxonomic clade.
    :param ofile_dir: The temp file (!) that contains a matrix with the taxa, their data, and labels.
    :return: 'ig_results_rounded' A dictionary with features and their IG values.
             'ig_mean_score' is the mean of the computed IG scores in 'ig_results_rounded'.
    """
    header_list = list(df_data.columns.values)
    target_name_col = "label"
    data_cols = header_list[0:len(header_list) - 1]

    ig_results = {}

    for col in data_cols:
        info_gain = calculate_information_gain(data=df_data, split_name=col, target_name=target_name_col)
        ig_results[col] = info_gain

    K = _DECIMAL_ROUND  # How many K decimal values should we round off to?
    ig_results_rounded = {key: round(ig_results[key], K) for key in ig_results}

    ig_mean_score = sum(ig_results_rounded.values()) / len(ig_results_rounded)

    return ig_results_rounded, ig_mean_score


def split_list(dataset, n):
    #   Split a list to sublists
    return [dataset[i:i + n] for i in list(range(0, len(dataset), n))]


def estimate_optimal_threshold(feature_records, child_parent_dict, leaf_name_set, taxon_value_dict, root_node):
    """
    Estimate optimal correlation threshold based on dataset characteristics. This function analyzes the correlation
    structure of taxonomic features to automatically determine an appropriate threshold for Stage 1 filtering in
    the HFE algorithm. The logic uses INVERTED threshold selection:
        - High median correlations → Lower threshold (more aggressive filtering)
        - Low median correlations → Higher threshold (less aggressive filtering)
    This inverted approach is counterintuitive but necessary because when parent-child taxonomic features are
    highly correlated (0.84+), you need a lower threshold to actually filter out the redundant ones, otherwise
    Stage 1 keeps too many similar features, leading to poor discrimination in Stage 2. Conversely, when features
    show low correlation, a higher threshold preserves the valuable diversity needed for effective information gain.

    For example:
    - Dataset with median correlation 0.85 → threshold 0.65 (remove highly redundant features)
    - Dataset with median correlation 0.45 → threshold 0.80 (preserve feature diversity)

    Parameters:
    -----------
    feature_records : dict
        Dictionary containing feature abundance vectors and labels
    child_parent_dict : dict
        Mapping of child taxonomic nodes to their parents
    leaf_name_set : set
        Set of leaf node names in the taxonomic hierarchy
    taxon_value_dict : dict
        Dictionary mapping taxon names to their abundance vectors
    root_node : str
        Root node identifier (e.g., "k__Bacteria")

    Returns:
    --------
    float
        Adaptive correlation threshold optimized for the dataset's correlation structure

    Notes:
    ------
    - Samples up to 100 parent-child pairs to estimate correlation distribution
    - Falls back to 0.7 if no valid correlations can be computed
    - Only activates when default threshold (0.7) is used
    """

    # Calculate dataset diversity metrics
    num_samples = len(feature_records["label"])
    num_features = len(leaf_name_set)
    feature_diversity = num_features / num_samples if num_samples > 0 else 0

    # Sample a subset of correlations to estimate distribution
    sample_correlations = []
    sample_size = min(100, len(leaf_name_set))

    for i, leaf in enumerate(list(leaf_name_set)[:sample_size]):
        if leaf in child_parent_dict and not (root_node in leaf):
            child_tax = leaf
            parent_tax = child_parent_dict[child_tax]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr = pearsonr(taxon_value_dict[child_tax], taxon_value_dict[parent_tax])[0]
                if not np.isnan(corr):
                    sample_correlations.append(abs(corr))

    if len(sample_correlations) == 0:
        return 0.7  # Default fallback

    # INVERTED logic - high correlation = lower threshold
    median_corr = np.median(sample_correlations)

    if median_corr > 0.9:
        adaptive_threshold = 0.6   # Very correlated data - be less aggressive
    elif median_corr > 0.8:
        adaptive_threshold = 0.65  # Moderately correlated - moderate filtering
    elif median_corr > 0.6:
        adaptive_threshold = 0.7   # Standard correlation
    else:
        adaptive_threshold = 0.8   # Low correlation data - be more aggressive

    print(f"[{get_print_time()}] Dataset correlation analysis:")
    print(f"[{get_print_time()}]  - Median correlation: {median_corr:.3f}")
    print(f"[{get_print_time()}]  - Adaptive threshold: {adaptive_threshold:.3f}")

    return adaptive_threshold


def verify_and_align_sample_order(abundance_file, labels_file, sample_ids_file):
    """
    Verify that all input files have the same number of samples in the same order.
    Since the original HFE code assumes this, we just verify sample counts match.
    """
    # Load data
    abundance_data = np.loadtxt(abundance_file, delimiter='\t', dtype=object)
    labels_data = np.loadtxt(labels_file, delimiter='\t', dtype=object)
    sample_ids_data = np.loadtxt(sample_ids_file, delimiter='\t', dtype=object)

    # Count samples in each file
    num_abundance_samples = abundance_data.shape[1] - 1  # Subtract 1 for feature name column
    num_label_samples = len(labels_data) - 1  # Subtract 1 for "label" header
    num_sample_id_samples = len(sample_ids_data) - 1  # Subtract 1 for "sample_id" header

    # Verify sample counts match
    if not (num_abundance_samples == num_label_samples == num_sample_id_samples):
        raise ValueError(f"Sample count mismatch: abundance={num_abundance_samples}, "
                         f"labels={num_label_samples}, sample_ids={num_sample_id_samples}")

    print(f"[{get_print_time()}]    Sample counts verified: {num_abundance_samples} samples")


def compute_hfe(dataset_name, abundance_file, labels_file, lineages_file, sample_ids_file, ig_method, output_dir,
                verbose=False, debug=False):
    """
    Executes the main HFE algorithm from the paper. Code adapted from the original implementation.

    :param dataset_name: The name of the current dataset we are using.
    :param abundance_file: Tab-delimited file containing the OTU abundances.
    :param labels_file: Single Row file (tab delim) with the labels for each sample in the OTU matrix.
    :param lineages: A taxonomic lineage file that contains the hiearchical clades for each microbe in the sample.
    :param ig_method: The method to use to compute the information gain (IG).
    :param output_dir: The base output directory.
    :param verbose: Flag for wordy terminal output.
    :param debug: Flag for even more terminal output, and saving temporary files.
    """

    print(f"[{get_print_time()}]")
    print(f"[{get_print_time()}] Starting HFE...")
    print(f"[{get_print_time()}]")

    corr_method = 1  # 1: pearson correlation

    root_node = ''
    tax_type = '0'
    if tax_type == '0':
        root_node = "k__Bacteria"

    # ----------------------------------------------- Data Loading ----------------------------------------------------
    #
    print(f'[{get_print_time()}] Dataset Name: {dataset_name}')
    print(f'[{get_print_time()}]  - IG Method: {ig_method.value}')

    # Verify sample ordering across all input files
    print(f"[{get_print_time()}]  - Verifying sample counts...")
    verify_and_align_sample_order(abundance_file, labels_file, sample_ids_file)

    print(f"[{get_print_time()}]  - Loading OTU Abundance Table... ", end="")

    # OTU Matrix. Creates a dictionary (each column is a sample, each row is a feature (e.g. an OTU)).
    dataset = np.loadtxt(abundance_file, delimiter='\t', dtype=object)
    feature_records = {str(line[0]): [float(i) for i in line[1:]] for line in dataset}

    print(f'Done. ({len(feature_records):,})')

    # Labels
    labels = np.loadtxt(labels_file, delimiter='\t', dtype=object)
    label_vector = {labels[0]: list(labels[1:])}
    feature_records.update(label_vector)

    # Sample Ids
    sample_ids = np.loadtxt(sample_ids_file, delimiter='\t', dtype=object)
    sample_id_vector = {sample_ids[0]: list(sample_ids[1:])}

    # Lineages. Create a dictionary of the lineages taxonomy file.
    tax_data = np.loadtxt(lineages_file, delimiter='\t', dtype=object)
    tax_file = {line[0]: list(line[1:]) for line in tax_data}
    tax_file_ordered = OrderedDict(sorted(tax_file.items(), key=lambda t: t[0]))

    # assuming that the columns are for each level in the hierarchy in addition to the leafs' labels.
    no_of_levels = len(tax_file[list(tax_file.keys())[0]])

    print(f"[{get_print_time()}]  - Number of Taxonomic Levels: {no_of_levels}")

    #   What will indicate an empty taxonomic level?
    #   This is used to test for things like "f__", "g__", or "s__".
    empty = []
    if tax_type == '0':
        empty = [3, 3, 3, 3, 3, 3, 3, 3, 3]

    child_parent_dict = {}
    leaf_name_set = set()
    taxon_value_dict = {}
    otu_parent_dict = {}

    print(f"[{get_print_time()}]  - Parsing Hierarchical Lineage file... ", end="")

    # parsing the tax file and prepare the data
    # for element in tax_file:
    for element in tax_file_ordered:

        tax = tax_file[element]  # assume that the line starts with the leaf label (e.g. otu_id) then the path details.

        # set the name of every leaf (i.e. labeling each leaf in the hierarchy with otu_id for example)
        if root_node in tax:
            leaf_indx = -1

            #   Where is the first occurrence of a "f__", "g__", or "s__" is located in the lineage array.
            if tax_type == '0':
                for i in range(1, no_of_levels):
                    if len(tax[i].strip()) == empty[i]:
                        leaf_indx = i - 1
                        break

            if leaf_indx == -1:
                leaf_indx = no_of_levels - 1

            if not (element in leaf_name_set):
                otu_parent_dict.update({element: tax[leaf_indx].strip()})
                taxon_value_dict.update({element: feature_records[element]})
                if tax_type == '0':
                    if 's__' in tax[leaf_indx].strip():  # If the OTU is the leaf, add the OTU to the leaf set.
                        leaf_name_set.add(element)
                        child_parent_dict.update({element: tax[leaf_indx].strip()})
                    else:
                        #   Otherwise, add whichever taxonomic level is the first occurrence of "[pcofg]__".
                        leaf_name_set.add(tax[leaf_indx].strip())

            for i in range(1, no_of_levels):
                # for each level, specify the parent of each child (assuming each child has one parent)
                if tax_type == '0':
                    if not (tax[i].strip() in child_parent_dict.keys()) and not (len(tax[i].strip()) == empty) and not (
                            len(tax[i - 1].strip()) == empty):
                        child_parent_dict.update({tax[i].strip(): tax[i - 1].strip()})

            # assign each taxon an abundance vector
            for i in range(no_of_levels):
                if not (tax[i].strip() in taxon_value_dict.keys()) and not (len(tax[i].strip()) == empty):
                    taxon_value_dict.update({tax[i].strip(): feature_records[element]})
                elif tax[i].strip() in taxon_value_dict.keys():
                    merged_values = list(np.sum([feature_records[element], taxon_value_dict[tax[i].strip()]], axis=0))
                    taxon_value_dict[tax[i].strip()] = merged_values

    print(f'Done. ({len(tax_file_ordered):,})')
    print(f"[{get_print_time()}]")


    # -------------------------------------------------- Output Prep --------------------------------------------------
    #
    #
    output_prefix = ""
    if ig_method == IGMethod.WEKA:
        output_prefix = IGMethod.WEKA.value
    elif ig_method == IGMethod.SCIKIT:
        output_prefix = IGMethod.SCIKIT.value
    elif ig_method == IGMethod.CUSTOM:
        output_prefix = IGMethod.CUSTOM.value

    # ---------------------------------------------------- Stage 1 ----------------------------------------------------
    #
    # The First phase of Hierarchy selection approach.
    #

    print(f"[{get_print_time()}] Stage 1...")

    # Estimate optimal threshold.
    corr_threshold = estimate_optimal_threshold(feature_records, child_parent_dict, leaf_name_set,
                                                taxon_value_dict, root_node)

    if debug:
        ofile1 = open(output_dir + f'{output_prefix}-correlations.tab', "wb")
        ofile1_str = 'Pair of taxons' + '\t' + "correlation coefficient" + '\n'
        ofile1.write(ofile1_str.encode())

    if debug:
        ofile2 = open(output_dir + f'{output_prefix}-removed_taxa.tab', "wb")
        ofile2_str = 'leaf' + '\t' + "taxon" + '\n'
        ofile2.write(ofile2_str.encode())

    removed_leaf_list = []
    leaves_to_add = []
    nodes_to_remove = []
    subtract_list = []

    leaf_name_list = list(leaf_name_set)
    leaf_name_list.sort()

    print(f"[{get_print_time()}]  - Filtering based on correlations...")
    print(f'[{get_print_time()}]  - Correlation Threshold: {corr_threshold:.2f}')

    for leaf in leaf_name_list:
        leaf_values = []
        if not (root_node in leaf):
            leaf_values = taxon_value_dict[leaf]

            # The leaf's taxon and parent (recall: assuming there is only one parent)
            child_tax = leaf
            parent_tax = ''
            if not (root_node in child_parent_dict[child_tax]):
                parent_tax = child_parent_dict[child_tax]

                # Correlation threshold
                # The aim is to remove child nodes that are redundant to their parents,
                # for which Pearson correlation serves as a proxy.
                correlation_threshold = corr_threshold

                similarity = 0.0

                # calculate the correlation between the child and parent
                if corr_method == 1:
                    # pearsonr' output is (correlation coefficient, p-value)
                    # The correlation coefficient is 'NAN' when either (or both) data sets are ZERO for all samples.
                    # When this happens, there is a 'PearsonRConstantInputWarning' thrown.
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        similarity = pearsonr(taxon_value_dict[child_tax], taxon_value_dict[parent_tax])[0]

                    if debug:
                        ofile1_str_2 = parent_tax + '_' + child_tax + '\t' + str(similarity) + '\n'
                        ofile1.write(ofile1_str_2.encode())

                # check the similarity value
                if (abs(similarity) > correlation_threshold) or (similarity == "nan"):
                    # remove the leaf
                    removed_leaf_list.append(child_tax)
                    nodes_to_remove.append(child_tax)
                    leaf_name_list.remove(child_tax)
                    leaf_name_set.remove(child_tax)

                    if debug:
                        ofile2_str_2 = leaf + '\t' + child_tax + '\n'
                        ofile2.write(ofile2_str_2.encode())

                    if not (parent_tax in leaf_name_set):
                        leaf_name_set.add(parent_tax)
                        leaf_name_list.append(parent_tax)

                        leaves_to_add.append(parent_tax)
                else:
                    # to subtract the abundance of the child from the parent's abundance
                    subtract_list.append((parent_tax, child_tax))
        else:
            leaf_name_list.remove(leaf)
            leaf_name_set.remove(leaf)
            nodes_to_remove.append(leaf)

    print(f"[{get_print_time()}]  - Number of leaves retained: {len(leaf_name_set):,}")
    print(f'[{get_print_time()}]')

    if debug:
        ofile3 = open(output_dir + f'{output_prefix}-leaves_phase1.tab', "wb")
        ofile3_str = 'leaf_name' + '\n'
        ofile3.write(ofile3_str.encode())

        for leaf in leaf_name_set:
            ofile3_str_2 = leaf + '\n'
            ofile3.write(ofile3_str_2.encode())

    if debug:
        ofile3.close()
        ofile1.close()
        ofile2.close()

    # create the temp feature set
    feature_value_FS = []
    for node in leaf_name_set:
        feature_value_FS.append([node] + taxon_value_dict[node])
    feature_value_FS.append(["label"] + feature_records["label"])
    transposed_table = list(zip(*feature_value_FS))

    if debug:
        ofile_dir = output_dir + f'{output_prefix}-temp_feature_set1.csv'
        into_tab(np.array(transposed_table), ofile_dir, ',')

    # Backup Stage 1 results before Stage 2 filtering
    stage1_survivors = leaf_name_set.copy()

    # ---------------------------------------------------- Stage 2 ----------------------------------------------------
    #
    # The Second phase of Hierarchy selection approach.
    #

    print(f'[{get_print_time()}] Stage 2...')
    print(f'[{get_print_time()}]  - Starting Java JVM...')

    loader = None
    if ig_method == IGMethod.WEKA:
        jvm.start(max_heap_size="1024m", packages=True)
        loader = Loader(classname="weka.core.converters.CSVLoader")

    path_dict = {}
    nodes_to_keep = []
    IG_dict = {}

    print(f'[{get_print_time()}]  - Computing information gain (IG) for nodes... ')

    # not to print a feature more than once
    IG_phase1_features = []
    phases_features = {}

    # [OLD CODE] Temp file used to pass data to WEKA functions.
    tmp_table_s2 = output_dir + f'{output_prefix}-temp_table.csv'

    for leaf in leaf_name_set:
        leaf_path = []
        leaf_taxon = leaf

        if leaf_taxon in child_parent_dict.keys():
            leaf_path.append(leaf_taxon)

            parent_tax = child_parent_dict[leaf_taxon]
            leaf_path.append(parent_tax)

            for i in range(no_of_levels):
                if leaf_path[-1] in child_parent_dict.keys():
                    if not (child_parent_dict[leaf_path[-1]] in leaf_path):
                        parent_tax = child_parent_dict[leaf_path[-1]]
                        leaf_path.append(parent_tax)
                else:
                    break

            # the active nodes in the path from the leaf to the root
            selected_node_LP = []
            feature_value_LP = []

            for n in leaf_path:
                if (n in leaf_name_set) and (not (root_node in n)):
                    selected_node_LP.append(n)
                    feature_value_LP.append([n] + taxon_value_dict[n])

            if len(selected_node_LP) > 0:
                # to compute the information gain via weka
                feature_value_LP.append(["label"] + feature_records["label"])
                transposed_table = list(zip(*feature_value_LP))

                if ig_method == IGMethod.WEKA:
                    into_tab(np.array(transposed_table), tmp_table_s2, ',')

                # New code uses a pandas df to send to the function, instead of the CSVs.
                tmp_data = np.array(transposed_table)
                df_tmp = pd.DataFrame(data=tmp_data[1:, :],    # All values from the first row onwards.
                                      columns=tmp_data[0, :])  # 1st row as the column names.
                cols = df_tmp.columns.difference(['label'])
                df_tmp[cols] = df_tmp[cols].astype(float)  # Cast all columns to float64, except for the label.

                results = {}
                mean_IG = 0.0

                if ig_method == IGMethod.WEKA:
                    results, mean_IG = get_IG(tmp_table_s2, loader=loader)
                elif ig_method == IGMethod.SCIKIT:
                    results, mean_IG = get_mutual_info(tmp_table_s2)
                elif ig_method == IGMethod.CUSTOM:
                    results, mean_IG = information_gain(df_tmp)

                if debug:
                    print(f'[{get_print_time()}] *************************************************')
                    print(f'[{get_print_time()}]  - Mean IG: {mean_IG}')
                    print(f'[{get_print_time()}]  - Filtering nodes with IG less than mean...')

                # check the IG score of each selected node in the path
                for s in selected_node_LP:
                    if not (s in IG_phase1_features):
                        IG_phase1_features.append(s)
                        phases_features.update({s: str(results[s])})

                    if (results[s] < mean_IG) or (results[s] <= 0):
                        nodes_to_remove.append(s)
                    else:
                        t = results[s]
                        nodes_to_keep.append(s)
                        if not (s in IG_dict.keys()):
                            # save the survived nodes with their IG scores
                            IG_dict.update({s: results[s]})
                            phases_features.update({s: str(t)})

    # remove what needs to be removed
    for r in nodes_to_remove:
        if not (r in nodes_to_keep) and (r in leaf_name_set):
            leaf_name_set.remove(r)

    # Early warning for empty IG_dict
    if len(IG_dict) == 0:
        print(f'[{get_print_time()}]  - WARNING: No features passed IG filtering in Stage 2')
        print(f'[{get_print_time()}]  - Consider lowering correlation threshold (current: {corr_threshold})')

    # print("Number of left features from phase 2: " + str(len(leaf_name_set)))
    print(f'[{get_print_time()}]  - Number of features retained: {len(leaf_name_set):,}')
    print(f'[{get_print_time()}]')

    # create the temp feature set
    feature_value_FS = []
    for node in leaf_name_set:
        if not (root_node in node):
            feature_value_FS.append([node] + taxon_value_dict[node])
    feature_value_FS.append(["label"] + feature_records["label"])
    transposed_table = list(zip(*feature_value_FS))

    if debug:
        ofile_dir = output_dir + f'{output_prefix}-temp_feature_set2.csv'
        into_tab(np.array(transposed_table), ofile_dir, ',')


    # ---------------------------------------------------- Stage 3 ----------------------------------------------------
    #
    #   The third phase of Hierarchy selection approach.
    #

    print(f'[{get_print_time()}] Stage 3...')
    print(f'[{get_print_time()}]  - IG-based leaf filtering...')

    # Check if any features survived Stage 2. If none survived, then `IG_dict` is empty.
    # This happens when:
    #  - High correlation threshold removes too many features in Stage 1.
    #  - All remaining features have IG ≤ mean_IG in Stage 2.
    #  - Small datasets with limited taxonomic diversity.
    #  - Highly correlated taxonomic structure where parent-child relationships are too similar.
    #
    otus_to_keep = {}
    feature_value_LP = []
    valid_otu_list = []
    IG_results = {}
    phase3_features = []
    IG_threshold = 0.0

    if len(IG_dict) == 0:
        print(f'[{get_print_time()}] WARNING: No features survived Stage 2 IG filtering.')
        print(f'[{get_print_time()}] Using remaining leaf features from Stage 1 instead.')
        # Use features that survived Stage 1 but failed Stage 2
        for leaf in stage1_survivors:
            if not (root_node in leaf):
                otus_to_keep[leaf] = 0.001  # Assign minimal but positive IG score
                phase3_features.append(leaf)
        print(f'[{get_print_time()}]  - Number of retained features: {len(phase3_features):,}')
    else:
        # calculate the avg. IG score of the nodes selected by the second phase
        IG_threshold = sum(IG_dict.values()) / len(IG_dict.values())

    for otu in otu_parent_dict.keys():
        if not (otu in nodes_to_remove) and not (otu in stage1_survivors):
            feature_value_LP.append([otu] + taxon_value_dict[otu])
            valid_otu_list.append(otu)

    # [OLD CODE] Temp file used to pass data to WEKA functions.. :(
    tmp_table_s3 = output_dir + f'{output_prefix}-temp_table.csv'

    # to optimize the time needed
    otu_table_partitions = split_list(feature_value_LP, 4)
    for partition in otu_table_partitions:
        if len(partition) != 0:
            partition.append(["label"] + feature_records["label"])

            transposed_table = list(zip(*partition))

            if ig_method == IGMethod.WEKA:
                into_tab(np.array(transposed_table), tmp_table_s3, ',')

            # New code uses a pandas df to send to the function, instead of the CSVs.
            tmp_data = np.array(transposed_table)
            df_tmp = pd.DataFrame(data=tmp_data[1:, :],    # All values from the first row onwards.
                                  columns=tmp_data[0, :])  # 1st row as the column names.
            cols = df_tmp.columns.difference(['label'])
            df_tmp[cols] = df_tmp[cols].astype(float)  # Cast all columns to float64, except for the label.

            results = {}
            avg_IG = 0.0

            if ig_method == IGMethod.WEKA:
                results, avg_IG = get_IG(tmp_table_s3, loader=loader)
            elif ig_method == IGMethod.SCIKIT:
                results, avg_IG = get_mutual_info(tmp_table_s3)
            elif ig_method == IGMethod.CUSTOM:
                # results, avg_IG = information_gain(tmp_table_s3)
                results, avg_IG = information_gain(df_tmp)

            IG_results.update(results)

    if ig_method == IGMethod.WEKA or ig_method == IGMethod.SCIKIT:
        os.remove(tmp_table_s3)  # This file is a tmp file used to pass data to functions, not used in CUSTOM code.

    # --------------------------------------------------- Stage 4 -------------------------------------------------------
    #   WARNING! — This seems to be Phase 4 from the paper. But unclear.
    #   Check the IG score of the selected otu against the overall avg. IG
    for s in IG_results:
        if (IG_results[s] > (1.0 * IG_threshold)) and (IG_results[s] > 0):
            otus_to_keep.update({s: IG_results[s]})
            phase3_features.append(s)
            phases_features.update({s: str(IG_results[s])})

    print(f'[{get_print_time()}]  - Number of retained OTUs: {len(phase3_features):,}')

    feature_value_FS = []  # Create the temp feature set
    for node in phase3_features:
        feature_value_FS.append([node] + taxon_value_dict[node])
    feature_value_FS.append(["label"] + feature_records["label"])
    transposed_table = list(zip(*feature_value_FS))

    if debug:
        ofile_dir = output_dir + f'{output_prefix}-temp_feature_set3.csv'
        into_tab(np.array(transposed_table), ofile_dir, ',')

    # ----------------------------------------------------- Output ----------------------------------------------------
    #
    #
    print(f'[{get_print_time()}]')
    print(f'[{get_print_time()}] Stopping Java JVM...')
    if ig_method == IGMethod.WEKA:
        jvm.stop()

    print(f'[{get_print_time()}]')
    print(f'[{get_print_time()}] Preparing output files...')

    for otu in otus_to_keep.keys():  # Add informative OTUs
        leaf_name_set.add(otu)

    final_features = set()
    if len(IG_dict) == 0:
        for leaf in stage1_survivors:  # Use Stage 1 survivors when Stage 2 failed
            if not (root_node in leaf):
                final_features.add(leaf)
    else:
        for leaf in leaf_name_set:  # Use normal pipeline results
            if not (root_node in leaf):
                final_features.add(leaf)

    #   Save the final feature list with IG scores.
    if debug:
        ofile_IG_name = f'{output_prefix}-final_feature_list_with_IG_scores.tab'
        ofile_IG = open(output_dir + ofile_IG_name, "wb")
        ofile_IG_str = 'Feature' + '\t' + 'IG_score' + '\n'
        ofile_IG.write(ofile_IG_str.encode())

        for leaf in leaf_name_set:
            if not (root_node in leaf):
                ofile_IG_str_2 = leaf + '\t' + phases_features[leaf] + '\n'
                ofile_IG.write(ofile_IG_str_2.encode())
        ofile_IG.close()

    # Save the final list of features (leaves). This only contains the names.
    final_list_file_name = 'final_feature_list.tab'
    ofile = open(output_dir + final_list_file_name, "wb")
    ofile_f_str = 'leaf_name' + '\n'
    ofile.write(ofile_f_str.encode())

    for leaf in final_features:
        ofile_str_2 = leaf + '\n'
        ofile.write(ofile_str_2.encode())
    ofile.close()

    # -----------------------------------------------------------------------------------------------------------------
    #
    #   Final Feature Set
    #   A Matrix, with features as columns, rows as samples, and cells the value of the feature in the sample.
    #
    print(f'[{get_print_time()}] Collecting final feature set...')

    feature_value_FS = []
    for node in final_features:
        feature_value_FS.append([node] + taxon_value_dict[node])

    print(f'[{get_print_time()}] Number of features in final set: {len(feature_value_FS):,}')

    #   Prepare the final output.
    feature_value_FS.insert(0, ["sample_id"] + sample_id_vector[sample_ids[0]])
    feature_value_FS.append(["label"] + feature_records["label"])
    transposed_table = list(zip(*feature_value_FS))

    #   Extract the table header for the DataFrame.
    table_header = list(transposed_table.pop(0))

    #   Save a tab-delimited file with the retained features and their values in the samples.
    file_name_feature_set = 'final_feature_set'
    ofile_dir = f"{output_dir}{file_name_feature_set}.csv"
    into_tab(np.array(transposed_table), ofile_dir, ',')

    #   Save a dataframe of the retained features and their values in the samples.
    if debug:
        df_final_set = pd.DataFrame(transposed_table,
                                    columns=table_header)
        df_file_name = output_dir + f"/{file_name_feature_set}.pkl"
        df_final_set.to_pickle(df_file_name)

    if os.path.exists(output_dir):
        os.system("chmod -R 777 " + output_dir)


# ------------------------------------------------------ MAIN ---------------------------------------------------------
#
if __name__ == '__main__':
    """
    Main function. Re-implemented from the original python2 codebase.
    Note on taxonomic file: Make sure that there is no redundancy in the tax file (Original comment).
    """

    start_time_overall = time.time()

    # ------------------------------------------ Resource Locations ---------------------------------------------------
    #
    parser = argparse.ArgumentParser(description="Hierarchical Feature Engineering (HFE)", add_help=True)
    required_args = parser.add_argument_group(description='Required Arguments')
    required_args.add_argument("--name", required=True, type=str, metavar="data1", help="Name of the data set.")
    required_args.add_argument("--abundances", required=True, type=str, metavar="path", help="Path to abundance file.")
    required_args.add_argument("--labels", required=True, type=str, metavar="path", help="Path to the labels file.")
    required_args.add_argument("--sample_ids", required=True, type=str, metavar="path", help="Path to the sample IDs file.")
    required_args.add_argument("--lineages", required=True, type=str, metavar="path", help="Path to lineage file.")
    required_args.add_argument("--ig_method", required=True, type=str, metavar="'custom'", default="custom",
                               choices=["custom", "scikit", "weka"],
                               help="Information Gain Method ('custom', 'scikit', or 'weka').")
    required_args.add_argument("--output_dir", required=True, type=str, metavar="path",
                               help="Path to the base output directory.")
    optional_args = parser.add_argument_group(description='Optional Arguments')
    optional_args.add_argument("--corr_threshold", required=False, type=float, metavar='0.7', default=0.7,
                               help="Correlation Threshold for stage 1. (default: %(default)s)")
    optional_args.add_argument("--verbose", action="store_true", required=False, help="Wordy Terminal output.")
    optional_args.add_argument("--debug", action="store_true", required=False, help="Debug mode. Lots of output.")

    #   Before collecting the arguments, we check that something was passed, and if it wasn't, then we exit.
    if len(sys.argv) == 1:
        print("\nERROR: No arguments. See below for usage.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args(sys.argv[1:])

    dataset_name    = args.name
    abundance_file  = args.abundances
    labels_file     = args.labels
    sample_ids_file = args.sample_ids
    lineages_file   = args.lineages
    ig_method_args  = args.ig_method
    corr_threshold  = args.corr_threshold
    output_dir      = args.output_dir
    verbose         = args.verbose
    debug           = args.debug

    output_dir = f'{output_dir}/{dataset_name}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ig_method = None
    if ig_method_args == IGMethod.WEKA.value:
        ig_method = IGMethod.WEKA
    if ig_method_args == IGMethod.SCIKIT.value:
        ig_method = IGMethod.SCIKIT
    if ig_method_args == IGMethod.CUSTOM.value:
        ig_method = IGMethod.CUSTOM

    # Call the HFE function.
    compute_hfe(dataset_name=dataset_name, abundance_file=abundance_file, labels_file=labels_file,
                lineages_file=lineages_file, sample_ids_file=sample_ids_file, ig_method=ig_method, output_dir=output_dir,
                verbose=verbose, debug=debug)


    # -------------------------------------------- End of Main --------------------------------------------------------
    #
    print(f'[{get_print_time()}]')
    print(f"[{get_print_time()}] Done.")
    end_time_overall = time.time()
    overall_runtime = end_time_overall - start_time_overall
    print(f"[{get_print_time()}] Overall Runtime {timedelta(seconds=overall_runtime)}")
    print(f"[{get_print_time()}]")
