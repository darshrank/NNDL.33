# Dataset Description

## Source Dataset

This project uses audio samples from the **Coswara Dataset**, an open-access respiratory audio dataset developed by the Indian Institute of Science (IISc) LEAP Lab.

Original dataset repository:  
https://github.com/iiscleap/Coswara-Data

Coswara contains crowdsourced respiratory, cough, and speech audio recordings collected for research on sound-based screening of respiratory health conditions, including COVID-19. The dataset includes multiple audio categories such as breathing sounds, cough sounds, sustained vowel phonation, and counting speech samples.

## Original Coswara Audio Categories

The original Coswara dataset includes the following types of audio samples:

- Fast breathing
- Slow breathing
- Deep cough
- Shallow cough
- Sustained vowel sounds
- Fast counting
- Slow counting

The dataset also includes participant metadata such as age, gender, location, health status, symptoms, and comorbidities.

## Dataset Modification for This Project

For this project, we modified the original Coswara dataset into a binary audio classification dataset:

| Class | Description |
|---|---|
| `cough` | Audio samples containing cough sounds from the Coswara dataset |
| `non_cough` | Audio samples that do not contain cough sounds, including breathing, speech, and vowel recordings |

The goal of this modification was to train a lightweight cough detection model that can distinguish cough sounds from other human audio activity.

## Label Mapping

The original Coswara audio categories were mapped as follows:

| Original Coswara Category | Project Label |
|---|---|
| Deep cough | `cough` |
| Shallow cough | `cough` |
| Fast breathing | `non_cough` |
| Slow breathing | `non_cough` |
| Sustained vowel sounds | `non_cough` |
| Fast counting | `non_cough` |
| Slow counting | `non_cough` |

## Use in Edge Impulse

The modified dataset was uploaded to Edge Impulse and used to train an audio classification model for cough detection.

The Edge Impulse pipeline used the audio samples to extract acoustic features and train a machine learning model for binary classification:

```text
Input audio → Feature extraction → Model training → Cough / Non-cough prediction