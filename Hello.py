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

import streamlit as st
from streamlit.logger import get_logger

import tensorflow as tf
import numpy as np

from tensorflow import keras

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="House Price Predictor",
        page_icon="🛖",
    )

    st.write("# Welcome to 'House Price Pridictor' Streamlit Application! 👋")

    # st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Simplest Neural Network Model trained on a few house price examples based on the 
        'Hello World' of Neural Network.
        **👈 Select a demo from the sidebar** to see some examples
        of what Streamlit can do!
        ### Want to learn more?
        - Check out [Hello World of Neural Network_tf](https://www.linkedin.com/pulse/hello-world-neural-network-tensorflow-muhammad-arslan-gnxrf/?trackingId=0hlhIzCCSRKAWuw5pu0DPA%3D%3D)
        - Follow me on [LinkedIn](www.linkedin.com/comm/mynetwork/discovery-see-all?usecase=PEOPLE_FOLLOWS&followMember=mecxlan)
      """
    )

# Load the trained model
# GRADED FUNCTION: house_model
def house_model():
    ### START CODE HERE

    # Define input and output tensors with the values for houses with 1 up to 6 bedrooms
    # Hint: Remember to explictly set the dtype as float
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)

    # Down Scaling of 10000 units = 1 and 5000 units = 0.5
    ys = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

    # Define your model (should be a model with 1 dense layer and 1 unit)
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

    # Compile your model
    # Set the optimizer to Stochastic Gradient Descent
    # and use Mean Squared Error as the loss function
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # Train your model for 1000 epochs by feeding the i/o tensors
    model.fit(xs, ys, epochs=1000)

    ### END CODE HERE
    return model

# Get your trained model
model = house_model()


def predict_price(bedrooms):
  # Convert the number of bedrooms to a NumPy array
  bedrooms = np.array([bedrooms], dtype=float)

  # Make a prediction using the trained model
  prediction = model.predict(bedrooms)[0]

  return prediction

# Create a Streamlit app
st.title('House Price Prediction')

bedrooms = st.slider('Number of bedrooms', 1, 10, 7)
predicted_price = predict_price(bedrooms)

st.write('Predicted price:', predicted_price)


if __name__ == "__main__":
    run()
