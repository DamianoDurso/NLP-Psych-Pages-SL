from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
import time 
import os
import streamlit as st
import logging 

@st.cache_resource
def load_model_pfa():
    try:
        with st.spinner("Loading the model..."):
            model_name = os.environ.get('remote_model_path', 'intfloat/multilingual-e5-base')
            time.sleep(1)
            st.write("Model found!")
            time.sleep(1)
            model = SentenceTransformer(model_name)
            st.write("Model loaded!")
            return model
    except Exception as e:
        logging.error(f'Error while loading models/tokenizer: {e}')
        st.error('Error while loading models/tokenizer.')
        return None

# Define functions
def avg_cosine_matrix(constructs, items):
    all_elements = constructs + items
    cor_matrices = []
    
    for mod in range(1,2):
        model = st.session_state.model_pfa
        embeds = model.encode(all_elements)
        cos_tem = util.cos_sim(embeds, embeds)
        out = np.array(cos_tem)
        cor_matrices.append(out)
    
    cor_matrices = np.mean(cor_matrices, axis=0)
    return cor_matrices

def cronbach_alpha_from_loadings(loadings_matrix):
    num_factors = loadings_matrix.shape[1]
    
    if loadings_matrix.shape[0] < 1:
        return None
    else:
        alphas = []

        for i in range(num_factors):
            loading_factor = loadings_matrix[:, i]
            correlation_matrix = np.outer(loading_factor, loading_factor.T)
            np.fill_diagonal(correlation_matrix, 1)
            alpha = cronbach_alpha_from_correlation(correlation_matrix)
            alphas.append(alpha)

        return alphas

def cronbach_alpha_from_correlation(correlation_matrix):
    num_items = correlation_matrix.shape[0]
    
    if num_items < 2:
        return "Add at least one item"
    else:
        np.fill_diagonal(correlation_matrix, 0)
        rmean = (np.sum(correlation_matrix)) / (num_items * (num_items - 1))
        alpha_std = num_items * rmean / (1 + (num_items - 1) * rmean)
        return alpha_std

def mcdonald_omega_total(loadings_matrix):

    omega = []
    num_factors = loadings_matrix.shape[1]

    for fac in range(num_factors):
        communality = np.square(loadings_matrix[:, fac])
        sum_squared_loadings = np.sum(loadings_matrix[:, fac])
        sum_squared_errors = np.sum(1 - communality)
        omega_total = (sum_squared_loadings ** 2) / ((sum_squared_loadings ** 2) + sum_squared_errors)
        omega.append(omega_total)
    return omega


def process_constructs_and_items(constructs, items):
    """
    Processes the given constructs and items, computes cosine similarities, and calculates reliability metrics.
    
    Args:
    constructs (list): A list of construct names.
    items (list): A list of item names.
    pfa (object): An object with methods for computing cosine similarities, Cronbach's alpha, and McDonald's omega.

    Returns:
    final_alpha (pd.DataFrame): A DataFrame containing constructs, Cronbach's alpha, McDonald's omega, and average cosine similarities.

    This function performs the following steps:
    1. Initializes a DataFrame with items as the index and a column for items.
    2. Iterates over each construct to compute cosine similarities with items using `pfa.avg_cosine_matrix`.
    3. Stores the computed similarities in the DataFrame, excluding the diagonal similarity.
    4. Constructs a loading matrix from the DataFrame.
    5. Computes Cronbach's alpha and McDonald's omega using the loading matrix.
    6. Computes the average cosine similarity for each construct.
    7. Returns a DataFrame with constructs, Cronbach's alpha, McDonald's omega, and average cosine similarities.
    """

    if constructs and items:
        # Initialize DataFrame with items as index and an "Item" column
        results_df = pd.DataFrame(index=items, columns=["Item"])
        
        # Compute cosine similarities for each construct
        for construct in constructs:
            if construct.strip() != "":
                similarities = avg_cosine_matrix([construct], items)
                index = constructs.index(construct)
                results_df[construct] = np.delete(similarities[index], index)
        
        # Add items to the DataFrame
        results_df["Item"] = items
        
        # Create loadings matrix
        loadings_matrix = results_df.iloc[:, 1:].values
        
        # Initialize final DataFrame to store reliability metrics
        final_alpha = pd.DataFrame(index=constructs, columns=['Constructs'])
        final_alpha['Std. Alpha'] = cronbach_alpha_from_loadings(loadings_matrix)
        final_alpha['Omega'] = mcdonald_omega_total(loadings_matrix)
        final_alpha['Average Cos'] = results_df.iloc[:, 1:].mean(axis=0)
        final_alpha['Constructs'] = constructs
        
        return results_df, final_alpha