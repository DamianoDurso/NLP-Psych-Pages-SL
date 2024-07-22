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


# Define functions
def avg_cosine_matrix(constructs, items, models):
    all_elements = constructs + items
    cor_matrices = []

    for i, mod in enumerate(models):
        model = models[i]
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