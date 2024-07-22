import streamlit as st
import numpy as np
import pandas as pd
import generation as generation

# Page configuration
st.set_page_config(page_title='Psychometric Item Generation', page_icon="🧠", layout="wide")

# Initialize session state
if 'outputs' not in st.session_state:
    st.session_state.outputs = []

# Initialize session state for the model
if 'model' not in st.session_state:
    with st.spinner('Loading model...'):
        st.session_state.model = generation.load_model_gen()

def show_disclaimer():
    st.title("Welcome to Psicometrista's item generation module!")
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
    st.title("Generate Psychometric Items")
    
    st.markdown("""
        Use the input fields below to create items aimed at assessing a particular psychological construct. Adjust the settings to influence the generated items.
    """)
        
    col1, col2 = st.columns(2)
    with col1:
        prefix_input = st.text_input('Item Prefix (Optional)', '')
        construct_input = st.text_input('Construct to Measure', 'Pessimism')
        
    with col2:
        sampling_options = ['Greedy Search', 'Beam Search', 'Multinomial Sampling']
        sampling_input = st.radio('Sampling Strategy', options=sampling_options, index=2)
        
        if sampling_input == 'Greedy Search':
            num_beams = 1
            num_return_sequences = 1
            temperature = 1.0
            top_k = 0
            top_p = 1.0
        elif sampling_input == 'Beam Search':
            num_beams = st.slider('Number of Beams', 1, 10, 3)
            num_return_sequences = st.slider('Number of Returned Beams', 1, 10, 2)
            temperature = 1.0
            top_k = 0
            top_p = 1.0
        else:  # Multinomial Sampling
            num_beams = 1
            num_return_sequences = 1
            temperature = st.slider('Temperature', 0.1, 1.5, 1.0)
            top_k = st.slider('Top-k Sampling', 0, 1000, 40)
            top_p = st.slider('Top-p Sampling', 0.0, 1.0, 0.95)
    
    if st.button('Generate Items', key='generate_items'):
        if construct_input:
            # Store construct input in session state
#            if construct_input not in st.session_state.constructs:
#                st.session_state.constructs.append(construct_input)
                
            kwargs = {
                'num_return_sequences': num_return_sequences,
                'num_beams': num_beams,
                'do_sample': sampling_input == 'Multinomial Sampling',
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p
            }
            item_stems = generation.generate_items(construct_input, prefix_input, **kwargs)
            st.session_state.outputs.append({'construct': construct_input, 'item': item_stems})
        else:
            st.error("Please enter a construct to measure.")

def display_results():
    if st.session_state.outputs:
        st.header("Generated Items")
        df = pd.DataFrame(st.session_state.outputs).explode('item').reset_index(drop=True)
        st.dataframe(df, use_container_width=True)

    else:
        st.info("No items generated yet. Enter a construct and click 'Generate Items'.")

def main():
    if 'show_disclaimer' not in st.session_state:
        st.session_state.show_disclaimer = True

    if st.session_state.show_disclaimer:
        show_disclaimer()
    else:
        show_item_generation()
        display_results()

if __name__ == '__main__':
    main()
