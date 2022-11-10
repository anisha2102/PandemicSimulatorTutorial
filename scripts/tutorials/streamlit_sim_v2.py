import streamlit as st
import mpld3
import streamlit.components.v1 as components

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd 
import copy
import os

import pandemic_simulator as ps

st.set_page_config(layout='wide')

if "load_state" not in st.session_state:
     st.session_state.load_state = False 

st.markdown("<h1 style='text-align: center; color: black;'>PandemicSim Demo</h1>", unsafe_allow_html=True)
st.markdown("<hr style=height:2px;border:none;color:#333;background-color:#333; /> ", unsafe_allow_html=True)

regulation_0 = ps.env.PandemicRegulation(  # moderate restriction
    stay_home_if_sick=False,  
    wear_facial_coverings=False,
    location_type_to_rule_kwargs={
        ps.env.Office: {'lock': False},  # unlock office (if locked)
        ps.env.School: {'lock': False},
        ps.env.HairSalon: {'lock': False},
        ps.env.RetailStore: {'lock': False},
        ps.env.Bar: {'lock': False},
        ps.env.Restaurant: {'lock': False},
    },
    stage=0  # a discrete identifier for this regulation
)

regulation_1 = ps.env.PandemicRegulation(  # moderate restriction
    stay_home_if_sick=True,  # stay home if sick
    wear_facial_coverings=False,
    location_type_to_rule_kwargs={
        ps.env.Office: {'lock': False},  # unlock office (if locked)
        ps.env.School: {'lock': False},
        ps.env.HairSalon: {'lock': False},
        ps.env.RetailStore: {'lock': False},
        ps.env.Bar: {'lock': False},
        ps.env.Restaurant: {'lock': False},
    },
    stage=1  # a discrete identifier for this regulation
)
regulation_2 = ps.env.PandemicRegulation(  # restricted movement
    stay_home_if_sick=True,  # stay home if sick
    wear_facial_coverings=True,
    location_type_to_rule_kwargs={
        ps.env.Office: {'lock': False},  # unlock office (if locked)
        ps.env.School: {'lock': True},
        ps.env.HairSalon: {'lock': True},
        ps.env.RetailStore: {'lock': False},
        ps.env.Bar: {'lock': False},
        ps.env.Restaurant: {'lock': False},
    },
    stage=2  # a discrete identifier for this regulation
)
regulation_3 = ps.env.PandemicRegulation(  # restricted movement
    stay_home_if_sick=True,  # stay home if sick
    wear_facial_coverings=True,
    location_type_to_rule_kwargs={
        ps.env.Office: {'lock': False},  # unlock office (if locked)
        ps.env.School: {'lock': True},
        ps.env.HairSalon: {'lock': True},
        ps.env.RetailStore: {'lock': False},
        ps.env.Bar: {'lock': True},
        ps.env.Restaurant: {'lock': True},
    },
    stage=3  # a discrete identifier for this regulation
)
regulation_4 = ps.env.PandemicRegulation(  # restricted movement
    stay_home_if_sick=True,  # stay home if sick
    wear_facial_coverings=True,
    location_type_to_rule_kwargs={
        ps.env.Office: {'lock': True},  # unlock office (if locked)
        ps.env.School: {'lock': True},
        ps.env.HairSalon: {'lock': True},
        ps.env.RetailStore: {'lock': True},
        ps.env.Bar: {'lock': True},
        ps.env.Restaurant: {'lock': True},
    },
    stage=4  # a discrete identifier for this regulation
)
regulation_map = {0: regulation_0, 1: regulation_1, 2:regulation_2, 3: regulation_3, 4: regulation_4}

ps.init_globals(seed=104923490)

sim_config = ps.sh.small_town_config

@st.experimental_singleton
def init():
    print("Initializing Environment")
    env = ps.env.PandemicGymEnv.from_config(sim_config = sim_config, pandemic_regulations=ps.sh.austin_regulations)
    gym_viz = ps.viz.GymViz.from_config(sim_config=sim_config)
    sim_viz = ps.viz.SimViz.from_config(sim_config=sim_config)
    print("Initialization Complete")
    env.reset()
    policy_env = copy.deepcopy(env)
    if os.path.exists("1.jpg"):
        os.remove("1.jpg")
    if os.path.exists("2.jpg"):
        os.remove("2.jpg")
    if os.path.exists("3.jpg"):
        os.remove("3.jpg")
    return env, sim_viz, gym_viz, policy_env


idx_to_label = ["Critical", "Dead", "Infected", "None", "Recovered"]


def simulate_days(days=10):
    print(f"Simulating {days} days")
    for _ in range(days):
        env.pandemic_sim.step_day()
        state = env.pandemic_sim.state
        sim_viz.record_state(state = env.pandemic_sim.state)
    print("Simulation Complete")


def simulate_summary(days=10, fname="dummy.jpg", regulation=None):
    if not regulation:
        env.pandemic_sim.impose_regulation(regulation_map[0])
    else:
        env.pandemic_sim.impose_regulation(regulation)
    simulate_days(days=days)

    fig1, axs1 = plt.subplots(ncols=2, nrows=2)
    plot=sim_viz.plot_critical_summary(ax=axs1[0,0])
    plot=sim_viz.plot_infection_source(ax=axs1[0,1])
    #plot=gym_viz.plot_cumulative_reward(ax=axs[1,0])
    plot=sim_viz.plot_stages(ax=axs1[1,0])
    plot=sim_viz.plot_gts(ax=axs1[1,1])

    fig1.tight_layout()
    fig1.set_figwidth(12)
    #st.pyplot(fig1, clear_figure=False)

    plt.savefig(fname)



def impose_regulation_and_simulate(stage):
    policy_env = copy.deepcopy(env)
    
    
    #env.pandemic_sim.impose_regulation(regulation_map[stage])
    print("Imposed Regulation",regulation_map[stage])

    simulate_summary(fname="2.jpg", regulation=regulation_map[stage])

def run_default_policy():
    """Here we execute the gym envrionment wrapped simulator using austin regulations,
    a small town config and default person routines."""

    print('\nA tutorial that runs the OpenAI Gym environment wrapped simulator', flush=True)

    #policy_env = env # ps.env.PandemicGymEnv(pandemic_sim = env, pandemic_regulations=ps.sh.austin_regulations)
    
    # setup viz
    #viz = ps.viz.GymViz.from_config(sim_config=sim_config)
    #sim_viz = ps.viz.SimViz.from_config(sim_config=sim_config)
    # run stage-0 action steps in the environment
    Reward = 0
    for i in range(10):
        if i==0: action=0
        if env.pandemic_sim.state.infection_above_threshold:
            action = min(action+1, 4)
        elif env.pandemic_sim.state.global_testing_state[...,2]<20:
            action = 0
        else:
            action = max(action-1, 0)


        obs, reward, done, aux = env.step(action=int(action))  # here the action is the discrete regulation stage identifier
        print(obs)
        Reward+=reward
        gym_viz.record((obs, reward))
        sim_viz.record_state(state = env.pandemic_sim.state)

    # generate plots
    print('Reward:'+str(Reward))

    fig, axs = plt.subplots(ncols=2, nrows=2)
    plot=sim_viz.plot_critical_summary(ax=axs[0,0])
    plot=sim_viz.plot_infection_source(ax=axs[0,1])
    #plot=gym_viz.plot_cumulative_reward(ax=axs[1,0])
    plot=sim_viz.plot_stages(ax=axs[1,0])
    plot=sim_viz.plot_gts(ax=axs[1,1])

    fig.tight_layout()
    fig.set_figwidth(12)
    #st.pyplot(fig)
    plt.savefig("3.jpg")



env, sim_viz, gym_viz, policy_env = init()
col1, col2 = st.columns((1, 2))
colInfo0 = st.columns(1)
st.markdown("<hr style=height:1px;border:none;color:#333;background-color:#333; /> ", unsafe_allow_html=True)

col3, col4 = st.columns((1, 1))
colInfo1, colInfo2 = st.columns((1, 1))

col5, col6 = st.columns((1, 1))

#tab1, tab2, tab3 = st.tabs(["Simulation", "Manual", "Default Policy"])

def display_images():
    if os.path.exists("1.jpg"): 
        with col2:
            image = Image.open('1.jpg')
            st.image(image, caption='Simulated Days',use_column_width=True)
    if os.path.exists("2.jpg"):
        
        with col5:
            image = Image.open('2.jpg')
            st.image(image, caption='',use_column_width=True)
    if os.path.exists("3.jpg"):
        
        with col6:
            image = Image.open('3.jpg')
            st.image(image, caption='',use_column_width=True)

with col1:
    with st.form("Simulate Form"):

        st.markdown("<h3 style='text-align: center; color: grey;'>Simulate Days</h1>", unsafe_allow_html=True)
        st.text("Simulate the start of the pandemic for 30 days.")
        simulate_button = st.form_submit_button(label='Simulate Days')
        if simulate_button:
            simulate_summary(30, fname="1.jpg")
            
            display_images()

            

        #st.button("Simulate Days", on_click=simulate_summary, args=[30])

with col3:   
     with st.form("Impose Regulation"):
        st.markdown("<h3 style='text-align: center; color: grey;'>Manual Regulation</h1>", unsafe_allow_html=True)
        stage = st.radio(
            "What stage regulation do you want to impose?",
            [0,1,2,3,4], horizontal=True, help="Stage 0: No locations are closed\n\nStage 1: No locations are closed, stay home is sick\n\nStage 2: Schools and Hair Salons are closed, stay home is sick, wear facial covernings\n\nStage 3: Schools, Hair Salons, Restaurants, and Bars are closed, stay home is sick, wear facial covernings\n\nStage 4: All locations are closed, stay home is sick, wear facial covernings")
        regulation_button = st.form_submit_button(label="Impose Regulation")
        if regulation_button:
            impose_regulation_and_simulate(stage)
            display_images()
            with colInfo1:
                if env.pandemic_sim.state.infection_above_threshold:
                
                    st.text("😨 Infection is above threshold!!!")
                else:
                    st.text("🙌 Infection is under control")



            print(env.observation)

        #st.button("Impose Regulation", on_click=impose_regulation_and_simulate, args=[stage])
with col4:
     with st.form("Policy Regulations"):
        st.markdown("<h3 style='text-align: center; color: grey;'>Policy Regulations</h1>", unsafe_allow_html=True)
        st.text("Run Policy to predict next Stage")
        st.text("This runs a default trained policy to predict the regulations")
        policy_button = st.form_submit_button(label="Run Policy")
        if policy_button:
            run_default_policy()
            display_images()
            with colInfo2:
                #st.text(f"Stages in past 10 days: {[x[...,0][0][0] for x in sim_viz._stages[-10:]]}")
                if env.pandemic_sim.state.infection_above_threshold:
                    st.text("😨 Infection is above threshold!!!")
                else:
                    st.text("🙌 Infection is under control")
   





