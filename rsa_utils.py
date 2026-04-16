import data_utils

from scipy.stats import pearsonr, kendalltau, spearmanr #type:ignore
from scipy.spatial.distance import cosine, pdist, squareform, euclidean #type:ignore
from sklearn.preprocessing import normalize #type:ignore
from sklearn.model_selection import KFold #type:ignore
from sklearn.metrics.pairwise import cosine_similarity #type:ignore
from sklearn.utils import shuffle #type:ignore
import matplotlib.pyplot as plt #type:ignore
import numpy as np #type:ignore
import statsmodels.api as sm #type:ignore


def basic_rsa(representations, Y, save_fig=True):

    ordered_indices = np.argsort(Y)[::-1]
    ordered_Y = Y[ordered_indices]
    ordered_representations = representations[ordered_indices]
    model_rdm = get_model_rdm(ordered_Y)
    data_rdm = get_rdm(ordered_representations, distance_metric='euclidean')
    result = correlate_rdms(model_rdm, data_rdm)
    return result.correlation, result.pvalue

def random_rsa(representations, Y):
    ordered_indices = np.argsort(Y)
    ordered_Y = Y[ordered_indices]
    ordered_representations = representations[ordered_indices]
    random_ground_truth = get_model_rdm(shuffle(Y))
    data_rdm = get_rdm(ordered_representations, distance_metric='euclidean')
    result = correlate_rdms(random_ground_truth, data_rdm)
    return result.correlation, result.pvalue




def correlation_and_rows(rdm_a, rdm_b, rdm_name, second_rdm_group_level_already = False, corr_metric = 'kendalltau', keep_corrs = False):
    corrs = []

    for group_i in range(40):
        if second_rdm_group_level_already:
            
            corr = correlate_rdms(data_utils.select_within_compound_groups(rdm_a, group_i), get_lower_triangle(rdm_b), correlation=corr_metric, select_lower_triangle=False)
        else:
            corr = correlate_rdms(data_utils.select_within_compound_groups(rdm_a, group_i), data_utils.select_within_compound_groups(rdm_b, group_i),
                                      correlation=corr_metric, select_lower_triangle=False)

        corrs.append(corr.correlation)

    corr_val = np.mean(corrs)
    std = np.std(corrs)

    
    row = {}

    row['{}_corr'.format(rdm_name)] = corr_val
    row['{}_std'.format(rdm_name)] = std

    if keep_corrs:
        return row, corrs
    else:
        return row
    
def correlation_and_rows_per_group(rdm_a, rdm_b, rdm_name, second_rdm_group_level_already=False, corr_metric='kendalltau', keep_corrs=False):
    corrs = []
    group_corr_dict = {}  # Dictionary to store per-group correlations

    for group_i in range(40):
        if second_rdm_group_level_already:
            
            corr = correlate_rdms(data_utils.select_within_compound_groups(rdm_a, group_i), get_lower_triangle(rdm_b), correlation=corr_metric, select_lower_triangle=False)
        else:
            corr = correlate_rdms(data_utils.select_within_compound_groups(rdm_a, group_i), data_utils.select_within_compound_groups(rdm_b, group_i),
                                      correlation=corr_metric, select_lower_triangle=False)


        corrs.append(corr.correlation)
        group_corr_dict[group_i] = corr.correlation  # Store correlation per group

    corr_val = np.mean(corrs)
    std = np.std(corrs)
    
    row = {}
    row['{}_corr'.format(rdm_name)] = corr_val
    row['{}_std'.format(rdm_name)] = std

    if keep_corrs:
        return row, corr, group_corr_dict  # Now returns a dictionary mapping group_i to correlation
    else:
        return row



def beta_values_rsa(X, Y):
    model = sm.OLS(Y, X).fit()
    # Get regression coefficients
    coefficients = model.params
    row = {}
    
    # save the coefficients as b1, b2, b3
    
    b1, b2, b3 = coefficients[1], coefficients[2], coefficients[3]
    row['b1'] = b1
    row['b2'] = b2
    row['b3'] = b3
    return row

def beta_values_rsa_per_group(X, Y):
    coefficients = []
    group_beta_dict = {}
    for group_i in range(40):
        model = sm.OLS(Y, X).fit()
    # Get regression coefficients
        coefficients.append(model.params)
    
        group_beta_dict[group_i] = model.params
    # save the coefficients as b1, b2, b3
    
    # b1, b2, b3 = coefficients[1], coefficients[2], coefficients[3]
    # row['b1'] = b1
    # row['b2'] = b2
    # row['b3'] = b3
    return group_beta_dict
    

def get_rdm(reps, distance_metric='euclidean'):
    return squareform(pdist(normalize(reps, axis=0), metric=distance_metric))

    
def get_normalized_rdm(reps, distance_metric='euclidean', save_normalized=True, save_path=None):
    # Normalize the representations
    normalized_reps = normalize(reps, axis=0)
    
    # Save normalized values if requested
    if save_normalized:
        if save_path is None:
            save_path = 'normalized_representations.npy'
        np.save(save_path, normalized_reps)
        print(f"Normalized representations saved to: {save_path}")
    
    return squareform(pdist(normalized_reps, metric=distance_metric))

def get_model_rdm(Y):
    return get_rdm(Y.reshape(-1, 1), distance_metric='euclidean')

def plot_mtx(mtx, title, figsize=(8,6)):
    plt.figure(figsize=figsize)
    plt.imshow(mtx, interpolation='nearest', cmap='Spectral_r')
    plt.title(title)
    plt.colorbar()

def get_lower_triangle(rdm):
    return rdm[np.where(np.triu(np.ones(rdm.shape)) == 0)]

def correlate_rdms(rdm_a, rdm_b, correlation="spearmanr", select_lower_triangle=True):
    corr_dict = {'spearmanr': spearmanr, 'kendalltau': kendalltau, 'pearsonr': pearsonr}
    corr_func = corr_dict[correlation]
    
    if select_lower_triangle:
        rdm_a = get_lower_triangle(rdm_a)
        rdm_b = get_lower_triangle(rdm_b)
    
    to_keep_inds = np.argwhere(~np.isnan(rdm_a) & ~np.isnan(rdm_b)).reshape(-1)
    new_rdm_a = rdm_a[to_keep_inds]
    new_rdm_b = rdm_b[to_keep_inds]
    
    return corr_func(np.array(new_rdm_a).flatten(), np.array(new_rdm_b).flatten())

# create a function that generates a mask for the outline region where i only want the bottom two rows of the first two columns

# def get_custom_outlined_mask(size=8):
#     """
#     Repeated-verb corner of each 8×8 group block: bottom two rows × first three columns.
#     Rows [size-2, size-1] × columns [0, 1, 2] (for size=8: rows 6–7, cols 0–2).
#     Clears the main diagonal; strict lower triangle only so symmetric RDMs are not double-counted.
#     """
#     mask = np.zeros((size, size), dtype=bool)
#     if size >= 2:
#         mask[size - 2 : size, 0:3] = True
#     np.fill_diagonal(mask, False)
#     return np.tril(mask, k=-1)

# def select_outlined_region(rdm, group_i, mask):
#     """
#     Extracts only the values from the RDM block that fall inside the mask.
#     """
#     start = group_i * 8
#     end = start + 8
#     # Extract the 8x8 block for the current group
#     block = rdm[start:end, start:end]
    
#     # Return only the values where the mask is True
#     return block[mask]


# def get_custom_outlined_mask(size=8):
#     """
#     Create a mask for the outlined region of the RDM.
#     """
#     mask_column = np.zeros((size, size), dtype=bool)
#     mask_column[:, 0:2] = True

#     mask_row = np.zeros((size, size), dtype=bool)
#     if size >= 2:
#         mask_row[size - 2 : size, :] = True

#     mask = mask_column | mask_row

#     np.fill_diagonal(mask, False)
#     return np.tril(mask, k=-1)


#### use this
def get_custom_outlined_mask(size=8):
    # 1. Initialize empty mask
    mask = np.zeros((size, size), dtype=bool)
    
    # 2. Define the top section of the shape (Rows 4-5, Cols 0-3)
    mask[4:6, 0:4] = True
    
    # 3. Define the bottom section of the shape (Rows 6-7, Cols 0-5)
    mask[6:8, 0:6] = True
    
    # 4. Remove diagonal and upper triangle to be safe 
    # (Though in an 8x8, these specific coordinates already avoid the diagonal)
    np.fill_diagonal(mask, False)
    return np.tril(mask, k=-1)

# def get_custom_outlined_mask(size=8):
#     """
#     Create a mask for the bottom 4 rows of the first 4 columns.
#     """
#     # 1. Initialize an 8x8 array of Falses
#     mask = np.zeros((size, size), dtype=bool)
    
#     # 2. Set the specific 4x4 block to True
#     # Rows 4, 5, 6, 7 and Columns 0, 1, 2, 3
#     mask[4:8, 0:4] = True

#     # 3. Standard RDM practice: 
#     # Ensure the diagonal and upper triangle remain False
#     np.fill_diagonal(mask, False)
#     return np.tril(mask, k=-1)




def select_outlined_region(rdm, group_i, mask):
    """
    Extracts only the values from the RDM block that fall inside the mask.
    """
    start = group_i * 8
    end = start + 8
    # Extract the 8x8 block for the current group
    block = rdm[start:end, start:end]
    
    # Return only the values where the mask is True
    return block[mask]

def correlation_outlined_only(rdm_a, rdm_b, rdm_name, second_rdm_group_level_already=False, corr_metric='kendalltau', keep_corrs=False):
    mask = get_custom_outlined_mask(8)
    corrs = []
    for group_i in range(40):
        # Extract only the outlined pixels from Model RDM
        vals_a = select_outlined_region(rdm_a, group_i, mask)
        
        # Extract from Target RDM
        if second_rdm_group_level_already:
            # If target is already a template (8x8), just apply mask
            vals_b = rdm_b[mask]
        else:
            vals_b = select_outlined_region(rdm_b, group_i, mask)

        # Calculate correlation on these specific vectors
        if len(vals_a) > 0:
            # Using your existing correlate_rdms logic but with flattened sub-selections
            c = correlate_rdms(vals_a, vals_b, correlation=corr_metric, select_lower_triangle=False)
            corrs.append(c.correlation)
    
    row = {}
    row['{}_corr'.format(rdm_name)] = np.mean(corrs)
    row['{}_std'.format(rdm_name)] = np.std(corrs)

    if keep_corrs:
        return row, corrs
    else:
        return row