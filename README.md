# Multi-class-classification-using-Trasformers-for-medical-text-
Medical data is extremely hard to find due to HIPAA privacy regulations. The dataset offers a solution by providing medical transcription samples. The dataset contains sample medical transcriptions for various medical specialties. This data was scraped from mtsamples.com. The aim of the project is to correctly classify the medical specialties based on the transcription text. 

The project is divided in sections: 

1. EDA (Understand the data, clean target variables, merge or rename categories) 
2. Labels visualization
3. Imbalance handling (we can make 2 models, one with weighted loss only and one with data agumentation + weighted loss)
4. Use pre-trained trasformers like ClinicalBERT (hugging face)
5. Training, Validation and Results visualization



This seems like a Multiclass problem rather than multi-label problem. 
  
<img width="406" height="204" alt="immagine" src="https://github.com/user-attachments/assets/51784b4f-6b99-4b2f-8117-c427efe1a189" />
