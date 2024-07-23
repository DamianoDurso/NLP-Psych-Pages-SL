import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
#import pseudo_fa as pfa
import pseudo_fa_2 as pfa

# Page configuration
st.set_page_config(page_title='Validate_items', page_icon="🎯", layout="wide")

# Initialize session state for constructs
if 'constructs' not in st.session_state:
    st.session_state.constructs = []

# Initialize session state for items
if 'items' not in st.session_state:
    st.session_state.items = []

# Initialize session state for the model
if 'model_pfa' not in st.session_state:
    with st.spinner('Loading model...'):
        st.session_state.model_pfa = pfa.load_model_pfa()



def show_disclaimer():
    st.title("Welcome to Psicometrista's item validation module!")
    st.subheader("Please read the disclaimer before using the app.")
    st.markdown("""
        **Disclaimer**:
        This application is provided as-is, without any warranty or guarantee of any kind, expressed or implied. It is intended for educational, non-commercial use only.
        The developers of this app shall not be held liable for any damages or losses incurred from its use. By using this application, you agree to the terms and conditions
        outlined herein and acknowledge that any commercial use or reliance on its functionality is strictly prohibited.
    """, unsafe_allow_html=True)
    if st.button('Accept Disclaimer', key='accept_disclaimer'):
        st.session_state.show_disclaimer = False

def show_item_generation():
    st.title("Validate Psychometric Items")
    st.subheader("Select Input Method")
    
def upload_data():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(df)
        if 'construct' in df.columns and 'item' in df.columns:
            st.session_state.constructs = df['construct'].tolist()
            st.session_state.items = df['item'].tolist()
        else:
            st.error("The CSV file must contain 'construct' and 'item' columns.")

def text_input_method():    

    constructs_input = st.sidebar.text_area("Enter Constructs (one per line)", key="constructs")
    items_input = st.sidebar.text_area("Enter Items (one per line)", key="items")
    if st.sidebar.button("Assess NLPsychometric Properties"):
        constructs = constructs_input.split("\n")
        items = items_input.split("\n")
        # Ensure non-empty constructs and items
        constructs = [construct.strip() for construct in constructs if construct.strip()]
        items = [item.strip() for item in items if item.strip()]

        #create function out of this     
        if constructs and items:
        #    results_df = pd.DataFrame(index=items, columns=["Item"])
        #    for construct in constructs:
        #        if construct.strip() != "":
        #            similarities = pfa.avg_cosine_matrix([construct], items)
        #            index = constructs.index(construct)
        #            results_df[construct] = np.delete(similarities[index], index)
#       # results_df = pfa.process_input(constructs, items)
#
        #    results_df["Item"] = items
        #    loadings_matrix = results_df.iloc[:, 1:].values
        #    final_alpha = pd.DataFrame(index=constructs, columns=['Constructs'])
        #    final_alpha['Std. Alpha'] = pfa.cronbach_alpha_from_loadings(loadings_matrix)
        #    final_alpha['Omega'] = pfa.mcdonald_omega_total(loadings_matrix)
        #    final_alpha['Average Cos'] = results_df.iloc[:, 1:].mean(axis=0)
        #    final_alpha['Constructs'] = constructs

            results_df, final_alpha = pfa.process_constructs_and_items(constructs, items)
            st.write("Pseudo Reliability Measures")
            st.dataframe(final_alpha)
        else:
            st.error("Please enter valid constructs and items.")

        try:
            #create function out of this
            cor_final, results_df_add = pfa.create_loading_matrix(results_df, constructs, items)


#            cor_mat = pfa.avg_cosine_matrix(constructs, items)
#            np.fill_diagonal(cor_mat, 1)
#            cor_mat =  cor_mat[len(constructs):, len(constructs):]
#            fa = FactorAnalyzer(n_factors=len(constructs), method='minres', rotation='oblimin', use_smc=True, is_corr_matrix=True).fit(cor_mat)
#            emp_load = fa.loadings_
#            emp_load = pd.DataFrame(fa.loadings_, index=results_df.index, columns=results_df.columns[1:]).add_suffix('_efa_loadings')
#            first_column = results_df.iloc[:, :1]
#            results_df_new = results_df.drop('Item', axis=1).add_suffix('_corr')
#            results_df_add = pd.concat([first_column, results_df_new, emp_load], axis=1)
        except Exception as e:
            st.write(f"FactorAnalyzer fitting failed: {e}")
            first_column = results_df.iloc[:, :1]
            results_df_new = results_df.drop('Item', axis=1).add_suffix('_corr')
            results_df_add = pd.concat([first_column, results_df_new], axis=1)
        
        st.write("Pseudo Loadings")
        st.write(results_df_add)

        st.write("Items Correlations")
        if len(items) > 1:
            headers = [item.split()[0] for item in items if item.split()]
            cor_item = cor_final[1:, 1:]
            fig, ax = plt.subplots()
            sns.heatmap(cor_item, yticklabels=headers, xticklabels=headers, annot=True, cmap='cividis', ax=ax)
            ax.set_title("Item Correlations", fontsize=10)
            st.pyplot(fig)

        st.write("Construct Correlations")
        if len(constructs) > 1:
            headers = [construct.split()[0] for construct in constructs if construct.split()]
            cor_cons = cor_final[:len(constructs), :len(constructs)]
            fig, ax = plt.subplots()
            sns.heatmap(cor_cons, yticklabels=headers, xticklabels=headers, annot=True, cmap='cividis', ax=ax)
            ax.set_title("Construct Correlations", fontsize=10)
            st.pyplot(fig)



def main():
    if 'show_disclaimer' not in st.session_state:
        st.session_state.show_disclaimer = True

    if st.session_state.show_disclaimer:
        show_disclaimer()
    else:
        show_item_generation()
        input_method = st.radio("Choose input method", ('Upload a dataset', 'Input text manually'))

        if input_method == 'Upload a dataset':
            upload_data()
        else:
            text_input_method()

if __name__ == '__main__':
    main()

# Initialize the model
#model_name = 'intfloat/multilingual-e5-base'
#model = SentenceTransformer(model_name)
#
#constructs_input = st.sidebar.text_area("Enter Constructs (one per line)", key="constructs")
#items_input = st.sidebar.text_area("Enter Items (one per line)", key="items")
#
#if st.sidebar.button("Assess NLPsychometric Properties"):
#    constructs = constructs_input.split("\n")
#    items = items_input.split("\n")
#
#    # Ensure non-empty constructs and items
#    constructs = [construct.strip() for construct in constructs if construct.strip()]
#    items = [item.strip() for item in items if item.strip()]
#    
#    if constructs and items:
#        results_df = pd.DataFrame(index=items, columns=["Item"])
#        for construct in constructs:
#            if construct.strip() != "":
#                similarities = pfa.avg_cosine_matrix([construct], items, model)
#                index = constructs.index(construct)
#                results_df[construct] = np.delete(similarities[index], index)
#        
#        results_df["Item"] = items
#        loadings_matrix = results_df.iloc[:, 1:].values
#        final_alpha = pd.DataFrame(index=constructs, columns=['Constructs'])
#        final_alpha['Std. Alpha'] = pfa.cronbach_alpha_from_loadings(loadings_matrix)
#        final_alpha['Omega'] = pfa.mcdonald_omega_total(loadings_matrix)
#        final_alpha['Average Cos'] = results_df.iloc[:, 1:].mean(axis=0)
#        final_alpha['Constructs'] = constructs
#
#        st.write("Pseudo Reliability Measures")
#        st.dataframe(final_alpha)
#
#        try:
#            cor_mat = pfa.avg_cosine_matrix(constructs, items, [model])
#            np.fill_diagonal(cor_mat, 1)
#            cor_mat = cor_mat[len(constructs):, len(constructs):]
#            fa = FactorAnalyzer(n_factors=len(constructs), method='minres', rotation='oblimin', use_smc=True, is_corr_matrix=True).fit(cor_mat)
#            emp_load = pd.DataFrame(fa.loadings_, index=results_df.index, columns=results_df.columns[1:]).add_suffix('_efa_loadings')
#            first_column = results_df.iloc[:, :1]
#            results_df_new = results_df.drop('Item', axis=1).add_suffix('_corr')
#            results_df_add = pd.concat([first_column, results_df_new, emp_load], axis=1)
#        except Exception as e:
#            st.write(f"FactorAnalyzer fitting failed: {e}")
#            first_column = results_df.iloc[:, :1]
#            results_df_new = results_df.drop('Item', axis=1).add_suffix('_corr')
#            results_df_add = pd.concat([first_column, results_df_new], axis=1)
#        
#        st.write("Pseudo Loadings")
#        st.dataframe(results_df_add)
#
#        st.write("Pseudo Construct Correlations")
#        if len(items) > 1:
#            headers = [item.split()[0] for item in items if item.split()]
#            cor_mat = pfa.avg_cosine_matrix(constructs, items, [model])
#            cor_mat = cor_mat[1:, 1:]
#            fig, ax = plt.subplots()
#            sns.heatmap(cor_mat, yticklabels=headers, xticklabels=headers, annot=True, cmap='cividis', ax=ax)
#            ax.set_title("Item Correlations", fontsize=10)
#            st.pyplot(fig)
#
#        st.write("Construct Correlations")
#        if len(constructs) > 1:
#            headers = [construct.split()[0] for construct in constructs if construct.split()]
#            cor_mat = pfa.avg_cosine_matrix(constructs, items, [model])
#            cor_mat = cor_mat[:len(constructs), :len(constructs)]
#            fig, ax = plt.subplots()
#            sns.heatmap(cor_mat, yticklabels=headers, xticklabels=headers, annot=True, cmap='cividis', ax=ax)
#            ax.set_title("Construct Correlations", fontsize=10)
#            st.pyplot(fig)
#