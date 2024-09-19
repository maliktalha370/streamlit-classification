import streamlit as st

def Lagopthalmos_detection_page():
    from lagophthalmos.app import run_app
    run_app()

def Bells_Detection_page():
    from bells.infer import run_app
    run_app()


def main_page():
    st.title("Welcome to the Eye Disease Detector")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.write("Choose which disease you want to detect:")
        lagop_button = st.button("Detect Lagophthalmos")
        bells_button = st.button("Detect Good / Bad Bells")

    if lagop_button:
        st.session_state.page = 'Lagopthalmos_detection'
    if bells_button:
        st.session_state.page = 'Bells_detection'

if 'page' not in st.session_state:
    st.session_state.page = 'main'

if st.session_state.page == 'main':
    main_page()
elif st.session_state.page == 'Lagopthalmos_detection':
    Lagopthalmos_detection_page()
elif st.session_state.page == 'Bells_detection':
    Bells_Detection_page()
