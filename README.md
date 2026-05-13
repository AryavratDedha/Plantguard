# PlantGuard — AI Crop Disease Detection

A two-stage CNN-based system that identifies crop type and diagnoses plant diseases from leaf images. Built using TensorFlow and MobileNet with transfer learning, deployed as a live web app.

## Live Demo

https://huggingface.co/spaces/dedhaaryavrat/Plant_Disease_Detection

## How It Works

The system works in two stages. First, a router model takes a healthy leaf image and identifies the crop type from eight supported crops. Second, a crop-specific disease model analyzes the affected leaf and returns a diagnosis along with confidence score and treatment recommendations.

## Supported Crops

Tomato, Apple, Corn, Potato, Rice, Sugarcane, Bell Pepper, Wheat

## Model Performance

Tomato disease model — 92.4% accuracy across 9 disease classes.
Router model — 99.4% accuracy across 8 crop types.

## Tech Stack

Python, TensorFlow, Keras, MobileNet, Transfer Learning, Streamlit

## Project Structure

app.py — Main Streamlit application
requirements.txt — Dependencies
models are hosted on Hugging Face due to file size

## About

This project was built independently as part of my ML learning journey. The goal was to go beyond standard tutorial projects and build something end-to-end — from data preprocessing and model training to deployment with a usable interface.

## Built By

Aryavrat Dedha, B.Tech Information Technology, BIET Jhansi
LinkedIn: linkedin.com/in/aryavrat-dedha
