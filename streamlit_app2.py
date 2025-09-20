import streamlit as st

st.header('st.slider')

# 単一の数値
age = st.slider('あなたの年齢を選んでください', 0, 100, 25)
st.write("あなたの年齢:", age)

# 範囲（タプル）
values = st.slider(
    '範囲を選んでください',
    0.0, 100.0, (25.0, 75.0)
)
st.write('選択した範囲:', values)
