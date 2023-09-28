import streamlit as st
import os
from PIL import Image
from RadModel import RadModel
import time

st.set_page_config(
    page_title="Covera Chatbot",
    page_icon="assets/logo.png" # ":robot_face:"
)
st.title("Covera Chatbot")


if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.input_image = None
    st.session_state.assistant_icon = Image.open("assets/assistant_icon.jpg")

if os.getenv("MODEL_TYPE") == "radfm":
    model = RadModel(os.getenv("MODEL_FOLDER"))
else:
    raise ValueError("MODEL_TYPE not supported")

def assistant_response(text: str, image: Image):
    res = model(question=text, ip_image=image)
    for t in res['answer'].split(' '):
        time.sleep(0.03)
        yield t + ' '


for message in st.session_state.messages:
    avatar = st.session_state.assistant_icon if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("image") is not None:
            st.image(message["image"], width=200)
        st.markdown(message["content"])

if prompt := st.chat_input("Add question here"):
    # Add user message to chat history
    msg = {"role": "user", "content": prompt}
    if st.session_state.input_image is not None:
        msg["image"] = st.session_state.input_image
    st.session_state.messages.append(msg)
    st.session_state.input_image = None
    # Display user message in chat message container
    with st.chat_message("user"):
        if msg.get("image") is not None:
            st.image(msg["image"], width=200, clamp=True)
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar=st.session_state.assistant_icon):
        message_placeholder = st.empty()
        full_response = ""
        if os.getenv("WITH_IMAGE") == "true" and msg.get("image") is None:
            full_response += "Please upload an image first."
        else:
            for response_chunk in assistant_response(prompt, msg.get("image")):
                full_response += response_chunk
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
        # Remove the cursor for the final display
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

if os.getenv("WITH_IMAGE") == "true":
    uploaded_file = st.file_uploader(
        "Upload", type=["jpg", "png"], label_visibility="collapsed"
    )
    if uploaded_file:
        img = Image.open(uploaded_file)
        if img is not None:
            st.image(img, width=400, clamp=True)
            st.session_state.input_image = img
    else:
        st.session_state.input_image = None

# print("--------------------------------------------------------------")
# model1 = RadModel(os.getenv("MODEL_FOLDER"))
# res = model1("describe this image", "./view1_frontal.jpg")
# print(f"*************   Prompt: {res['prompt']}, Answer: {res['answer']}  ***************")