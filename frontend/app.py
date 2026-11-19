import streamlit as st
import requests

st.set_page_config(
    page_title="Potato Leaf Disease Detector",
    page_icon="🥔",
    layout="centered"
)

st.title("🥔 Potato Leaf Disease Detection")
st.write("Upload a potato leaf image to detect its disease.")

# Backend API URL
API_URL = "https://potato-leaf-disease-detection-3x8q.onrender.com/predict"

# Image upload
uploaded_file = st.file_uploader(
    "Upload Potato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display uploaded image
    st.image(
        uploaded_file,
        caption="Uploaded Potato Leaf",
        use_container_width=True
    )

    if st.button("Predict Disease"):

        with st.spinner("Analyzing image..."):

            try:
                # Send image to FastAPI
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=60
                )

                response.raise_for_status()

                result = response.json()

                if "predicted_class" in result:

                    st.success("Prediction completed!")

                    # Display prediction
                    st.subheader("Prediction Result")

                    st.write(
                        "Disease:",
                        result["predicted_class"]
                    )

                    st.metric(
                        "Model Score",
                        f'{result["confidence"]}%'
                    )

                    # Display all class scores
                    st.subheader("All Class Predictions")

                    for name, score in result["all_scores"].items():

                        st.write(f"{name}: {score}%")

                        st.progress(
                            min(max(float(score) / 100, 0.0), 1.0)
                        )

            except requests.exceptions.RequestException as e:

                st.error(f"Backend API error: {e}")

else:
    st.info("Please upload a potato leaf image.")