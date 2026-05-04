import streamlit as st
import requests
import time
import matplotlib.pyplot as plt

URL_V1_SUM = "http://127.0.0.1:8000/api/summarize"
URL_V2_SUM = "http://127.0.0.1:8001/api/summarize"

URL_V1_QA = "http://127.0.0.1:8000/api/qa"
URL_V2_QA = "http://127.0.0.1:8001/api/qa"

st.set_page_config(layout="wide")
st.title("📊 Model Comparison Dashboard")

tab1, tab2 = st.tabs(["📄 Summarization", "❓ QnA"])

# ===================== SUMMARIZATION =====================
with tab1:
    st.header("Summarization Comparison")

    context = st.text_area("Enter Paragraph")

    if st.button("Summarize & Compare"):

        if not context.strip():
            st.warning("Enter text first")
        else:
            col1, col2 = st.columns(2)

            # Distil Models
            start = time.time()
            res1 = requests.post(URL_V1_SUM, json={"text": context}).json()
            t1 = round(time.time() - start, 4)
            sum1 = res1["summary"]

            # BART Models
            start = time.time()
            res2 = requests.post(URL_V2_SUM, json={"text": context}).json()
            t2 = round(time.time() - start, 4)
            sum2 = res2["summary"]

            input_len = max(len(context.split()), 1)
            len1 = len(sum1.split())
            len2 = len(sum2.split())

            comp1 = len1 / input_len
            comp2 = len2 / input_len

            # Display
            with col1:
                st.subheader("🔹 Distil Models")
                st.write(sum1)

            with col2:
                st.subheader("🔹 BART Models")
                st.write(sum2)

            # Metrics Table
            st.subheader("📊 Metrics")

            st.table({
                "Metric": ["Time", "Length", "Compression"],
                "Distil Models": [t1, len1, round(comp1, 3)],
                "BART Models": [t2, len2, round(comp2, 3)]
            })

            # Graph
            labels = ["Time", "Length", "Compression"]
            v1 = [t1, len1, comp1]
            v2 = [t2, len2, comp2]

            x = range(len(labels))
            width = 0.35

            plt.figure()
            plt.bar([i - width/2 for i in x], v1, width, label="Distil")
            plt.bar([i + width/2 for i in x], v2, width, label="BART")
            plt.xticks(x, labels)
            plt.legend()

            st.pyplot(plt)

            # Winner Logic
            score_v1 = (t1 < t2) + (len1 < len2)
            score_v2 = (t2 < t1) + (len2 < len1)

            winner = "Tie" if score_v1 == score_v2 else ("Distil Models" if score_v1 > score_v2 else "BART Models")
            st.success(f"🏆 Winner: {winner}")


# ===================== QNA =====================
with tab2:
    st.header("QnA Comparison")

    context = st.text_area("Enter Context", key="qa_context")
    question = st.text_input("Enter Question")

    if st.button("Answer & Compare"):

        if not context.strip() or not question.strip():
            st.warning("Enter both context and question")
        else:
            col1, col2 = st.columns(2)

            payload = {"question": question, "context": context}

            # Distil Models
            start = time.time()
            qa1 = requests.post(URL_V1_QA, json=payload).json()
            t1 = round(time.time() - start, 4)

            # RoBERTa Models
            start = time.time()
            qa2 = requests.post(URL_V2_QA, json=payload).json()
            t2 = round(time.time() - start, 4)

            ans1 = qa1["answer"]
            ans2 = qa2["answer"]

            len1 = len(ans1.split())
            len2 = len(ans2.split())

            # Display
            with col1:
                st.subheader("🔹 Distil Models")
                st.write(ans1)

            with col2:
                st.subheader("🔹 RoBERTa Models")
                st.write(ans2)

            # Metrics
            st.subheader("📊 Metrics")

            st.table({
                "Metric": ["Time", "Confidence", "Answer Length"],
                "Distil Models": [t1, qa1["confidence"], len1],
                "RoBERTa Models": [t2, qa2["confidence"], len2]
            })

            # Graph
            labels = ["Time", "Confidence", "Length"]
            v1 = [t1, qa1["confidence"], len1]
            v2 = [t2, qa2["confidence"], len2]

            x = range(len(labels))
            width = 0.35

            plt.figure()
            plt.bar([i - width/2 for i in x], v1, width, label="Distil")
            plt.bar([i + width/2 for i in x], v2, width, label="RoBERTa")
            plt.xticks(x, labels)
            plt.legend()

            st.pyplot(plt)

            # Winner Logic
            score_v1 = (t1 < t2) + (qa1["confidence"] > qa2["confidence"])
            score_v2 = (t2 < t1) + (qa2["confidence"] > qa1["confidence"])

            winner = "Tie" if score_v1 == score_v2 else ("Distil Models" if score_v1 > score_v2 else "RoBERTa Models")
            st.success(f"🏆 Winner: {winner}")