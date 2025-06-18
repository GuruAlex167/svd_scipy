import streamlit as st
import numpy as np
from PIL import Image
from scipy.linalg import svd
from io import BytesIO

st.title("📉 Image Compression with SVD")

st.markdown("""
Upload an image (JPG/PNG), convert it to grayscale, and compress it using **truncated SVD**.
Use the slider to select the rank `k` for compression.
""")

# Function to convert image to grayscale numpy array
@st.cache_data(show_spinner=False)
def load_and_process_image(uploaded_file):
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    return np.array(image), image

# Function to compute SVD
@st.cache_data(show_spinner=False)
def compute_svd(image_array):
    return svd(image_array, full_matrices=False)

# Function to compress image using k-rank approximation
def compress_image(U, S, VT, k):
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]
    compressed = np.dot(U_k, np.dot(S_k, VT_k))
    compressed = np.clip(compressed, 0, 255)
    return compressed.astype(np.uint8)

# Upload interface
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_array, original_image = load_and_process_image(uploaded_file)
    U, S, VT = compute_svd(image_array)

    st.subheader("Compression Slider")
    k = st.slider("Select compression rank (k)", min_value=5, max_value=min(image_array.shape)-1, value=50, step=5)

    compressed_image = compress_image(U, S, VT, k)

    # Display side-by-side
    st.subheader("Original vs Compressed Image")
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original (Grayscale)", use_column_width=True)
    with col2:
        st.image(compressed_image, caption=f"Compressed (k={k})", use_column_width=True)

    # Metrics
    st.subheader("📊 Compression Metrics")

    original_size = np.prod(image_array.shape)
    compressed_size = U[:, :k].size + S[:k].size + VT[:k, :].size
    compression_ratio = original_size / compressed_size
    reconstruction_error = np.linalg.norm(image_array - compressed_image, ord='fro')

    st.markdown(f"""
    - **Compression Ratio**: {compression_ratio:.2f}x  
    - **Reconstruction Error (Frobenius norm)**: {reconstruction_error:.2f}
    """)
else:
    st.info("Upload a JPG or PNG image to begin.")
