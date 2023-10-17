"""
Main script for DARKSKIES Bayesian neural network
"""
import pickle

with open('../data/binned_data_20.pkl', 'rb') as file:
    params, images = pickle.load(file)
