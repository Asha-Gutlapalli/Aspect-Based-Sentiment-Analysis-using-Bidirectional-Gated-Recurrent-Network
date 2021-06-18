import streamlit as st

# this caches the output to store the output and not call this function again
# and again preventing time wastage. `allow_output_mutation = True` tells the
# function to not hash the output of this function and we can get away with it
# because no arguments are passed through this.
# https://docs.streamlit.io/en/stable/api.html#streamlit.cache
@st.cache(allow_output_mutation=True, show_spinner=False)
def get_models():
  from absa import ABSA
  return {
    'ABSA' : ABSA()
  }

# load all the models before the app starts
with st.spinner('Loading Model...'):
  MODELS = get_models()

# description
st.markdown("# :heart_eyes: ABSA using Bi-GRU :rage:!")
st.write('''
ABSA is Aspect Based Sentiment Analysis which is a fine-grained Sentiment Analysis. \
It identifies both the aspect and the sentiment in the text. This is achieved using \
a sequential Recurrent Neural Network called the Bidirectional Gated Recurrent Unit. \
It is known for low memory consumption and high performance. This model predicts the \
aspect category and the sentiment class given a laptop review.
''')

# instruction
st.markdown(":bulb: Please write your laptop model name and review for the same. :bulb:")
model = MODELS['ABSA']

# laptop model
st.text_input("Name of Laptop Model", value='Hp Omen', key="Text")

# review
default_ = "But if you buy this i would highly recommend that you buy a case and screen protector for this machine as they are known to be quite fragile."
review = st.text_area("Review", value=default_, key="Text")

# predict aspect category and sentiment class
if st.button("Predict"):
  with st.spinner('Predicting...'):
    aspect_label, sentiment_label = model.predict(review)
  st.markdown("**:tada: Aspect Category: **" + aspect_label)
  st.markdown("**:tada: Sentiment Class: **" + sentiment_label)