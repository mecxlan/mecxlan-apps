# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import time

# import numpy as np

# import streamlit as st
# from streamlit.hello.utils import show_code


# def plotting_demo():
#     progress_bar = st.sidebar.progress(0)
#     status_text = st.sidebar.empty()
#     last_rows = np.random.randn(1, 1)
#     chart = st.line_chart(last_rows)

#     for i in range(1, 101):
#         new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
#         status_text.text("%i%% Complete" % i)
#         chart.add_rows(new_rows)
#         progress_bar.progress(i)
#         last_rows = new_rows
#         time.sleep(0.05)

#     progress_bar.empty()

#     # Streamlit widgets automatically run the script from top to bottom. Since
#     # this button is not connected to any other logic, it just causes a plain
#     # rerun.
#     st.button("Re-run")
import streamlit as st

# Set the page configuration
st.set_page_config(page_title="MRI Brain Tumor Segmentation", page_icon="🧠")

# The rest of your Streamlit app code goes here
st.markdown("# MRI Brain Tumor Segmentation")
# st.sidebar.header("Plotting Demo")
st.write(
    """ This MRI-Brain Tumor Segmentation illustrates my Final Year Project. It requires input MRI Image in JPEG or PNG format.
    It'll apply a Pre-Trained Model apply a CNN-Convolution Neural Network Named "deeplabv3_resnet50_random". Enjoy!"""
)

# Changes
import os
from os.path import splitext
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
import torchvision
import wget 


destination_folder = "output"
destination_for_weights = "weights"

if os.path.exists(destination_for_weights):
    print("The weights are at", destination_for_weights)
else:
    print("Creating folder at ", destination_for_weights, " to store weights")
    os.mkdir(destination_for_weights)
    
segmentationWeightsURL = 'https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/deeplabv3_resnet50_random.pt'

if not os.path.exists(os.path.join(destination_for_weights, os.path.basename(segmentationWeightsURL))):
    print("Downloading Segmentation Weights, ", segmentationWeightsURL," to ",os.path.join(destination_for_weights, os.path.basename(segmentationWeightsURL)))
    filename = wget.download(segmentationWeightsURL, out = destination_for_weights)
else:
    print("Segmentation Weights already present")

torch.cuda.empty_cache()

def collate_fn(x):
    x, f = zip(*x)
    i = list(map(lambda t: t.shape[1], x))
    x = torch.as_tensor(np.swapaxes(np.concatenate(x, 1), 0, 1))
    return x, f, i

model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, aux_loss=False)
model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[-1].kernel_size)

print("loading weights from ", os.path.join(destination_for_weights, "deeplabv3_resnet50_random"))

if torch.cuda.is_available():
    print("cuda is available, original weights")
    device = torch.device("cuda")
    model = torch.nn.DataParallel(model)
    model.to(device)
    checkpoint = torch.load(os.path.join(destination_for_weights, os.path.basename(segmentationWeightsURL)))
    model.load_state_dict(checkpoint['state_dict'])
else:
    print("cuda is not available, cpu weights")
    device = torch.device("cpu")
    checkpoint = torch.load(os.path.join(destination_for_weights, os.path.basename(segmentationWeightsURL)), map_location = "cpu")
    state_dict_cpu = {k[7:]: v for (k, v) in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict_cpu)

model.eval()

def segment(input):
    inp = input
    x = inp.transpose([2, 0, 1]) 
    x = np.expand_dims(x, axis=0)     
    
    mean = x.mean(axis=(0, 2, 3))
    std = x.std(axis=(0, 2, 3))
    x = x - mean.reshape(1, 3, 1, 1)
    x = x / std.reshape(1, 3, 1, 1)
    
    with torch.no_grad():
        x = torch.from_numpy(x).type('torch.FloatTensor').to(device)
        output = model(x)    
    
    y = output['out'].numpy()
    y = y.squeeze()
    
    out = y>0    
    
    mask = inp.copy()
    mask[out] = np.array([0, 0, 255])
    
    return mask

# import gradio as gr

# i = gr.inputs.Image(shape=(112, 112), label="Input Brain MRI")
# o = gr.outputs.Image(label="Hasil Segmentasi")

# examples = [["TCGA_CS_5395_19981004_12.png"], 
#             ["TCGA_CS_5395_19981004_14.png"],
#             ["TCGA_DU_5849_19950405_20.png"],
#             ["TCGA_DU_5849_19950405_24.png"],
#             ["TCGA_DU_5849_19950405_28.png"]]

# title = "Sistem Segmentasi Citra MRI Otak berbasis Artificial Intelligence"
# description = "This system is designed to help automate the process of accurately and efficiently segmenting brain MRIs into regions of interest. It does this by using a UBNet-Seg Architecture that has been trained on a large dataset of manually annotated brain images."

# article = "<p style='text-align: center'>Created by <a target='_blank' href='https://fi.ub.ac.id/'>Jurusan Fisika, FMIPA, Universitas Brawijaya </a></p>"


# gr.Interface(segment, i, o, 
#     allow_flagging = False, 
#     description = description,
#     title = title,
#     article = article,
#     examples = examples,
#     analytics_enabled = False).launch()
# plotting_demo()

# show_code(plotting_demo)
import streamlit as st
from PIL import Image
import numpy as np

def segment(image):
    # Your segmentation code here
    return segmented_image

st.title("Brain Tumor Segmentation Web Application")
st.write("This system is designed to help automate the process of accurately and efficiently segmenting brain MRIs into regions of interest. It does this by using a UBNet-Seg Architecture that has been trained on a large dataset of manually annotated brain images.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI.', use_column_width=True)
    if st.button('Segmentasi'):
        result = segment(np.array(image))
        st.image(result, caption='Hasil Segmentasi.', use_column_width=True)

