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

LOGGER = get_logger(__name__)

st.set_page_config(
   page_title="M.Arslan",
   page_icon="🅰️"
)

st.write("# Welcome to Muhammad Arslan's Streamlit Applications! 👋")

 
st.markdown(
        """
        Showcase portfolio & Practical Implementation of Learning.
        
        
        **👈 Select a demo from the sidebar** to see some examples of what Streamlit can do!        
        ### Want to learn more?

                
        - Check out [Hello World of Neural Network_tf](https://www.linkedin.com/pulse/hello-world-neural-network-tensorflow-muhammad-arslan-gnxrf/?trackingId=0hlhIzCCSRKAWuw5pu0DPA%3D%3D)
        - Follow me on [LinkedIn](www.linkedin.com/comm/mynetwork/discovery-see-all?usecase=PEOPLE_FOLLOWS&followMember=mecxlan)

      """
)
st.link_button("💻 Github", "https://github.com/mecxlan/") 
st.link_button("🧭 Data Analytics", "https://mecxlan.hashnode.dev/") 

st.link_button("📅 Data Sets", "https://www.kaggle.com/mecxlan") 
st.link_button("🔗 LinkedIn", "www.linkedin.com/comm/mynetwork/discovery-see-all?usecase=PEOPLE_FOLLOWS&followMember=mecxlan")

if __name__ == "__main__":
    run()
