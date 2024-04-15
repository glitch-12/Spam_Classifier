import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def main():
    # Set page config to full width and dark theme
    st.set_page_config(
        layout="wide",
        page_title="Spam Classifier App",
        page_icon=":fire:",
        initial_sidebar_state="expanded",
    )

    # Custom CSS to change background color to black
    st.markdown(
        """
        <style>
        body {
            background-color: #000000;
            color: #ffffff;
        }
        .st-dc {
            background-color: #E6E6FA;
            color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.title("Email/SMS Spam Classifier")
    # Page options
    page = st.radio("Choose a page", ["Home", "Messages", "Code"])

    if page == "Home":
        input_sms = st.text_area("Enter the message")

        if st.button('Predict'):
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
    
    elif page == "Messages":
        st.title("Messages for Spam and Not Spam")
        
        # Spam messages
        st.subheader("Spam Messages:")
        st.info("""
        1. Fancy a shag? I do.Interested? sextextuk.com txt XXUK SUZY to 69876. Txts cost 1.50 per msg. TnCs on website. X
        2. Congratulations! You've won $1,000,000!Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
        """)

        # Not Spam messages
        st.subheader("Not Spam Messages:")
        st.info("""
        1. Hey, how are you doing?
        2. Reminder: Meeting at 3 PM today.
        """)

    elif page == "Code":
        st.title("Streamlit App Code")
        st.code("""
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

def main():
    # Set page config to full width
    st.set_page_config(layout="wide")

    # Set background color to black
    st.markdown(
        '''
        <style>
        body {
            background-color: #000000;
            color: #ffffff;
        }
        .st-dc {
            background-color: #E6E6FA;
            color: #000000;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )

    # Page options
    page = st.radio("Choose a page", ["Home", "Messages", "Code"])

    if page == "Home":
        input_sms = st.text_area("Enter the message")

        if st.button('Predict'):
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
    
    elif page == "Messages":
        st.title("Messages for Spam and Not Spam")

        # Set background color to lavender for the message sections
        st.markdown(
            '''
            <style>
            .st-dc {
                background-color: #E6E6FA;
                color: #000000;
            }
            </style>
            ''',
            unsafe_allow_html=True
        )

        # Spam messages
        st.subheader("Spam Messages:")
        st.info(""
        1. Free Viagra now!!!
        2. Congratulations! You've won $1,000,000!
        "")

        # Not Spam messages
        st.subheader("Not Spam Messages:")
        st.info(""
        1. Hey, how are you doing?
        2. Reminder: Meeting at 3 PM today.
        "")

if __name__ == '__main__':
    main()
        """, language="python")

if __name__ == '__main__':
    main()

