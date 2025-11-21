
#%%  Libraries: 

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, get_linear_schedule_with_warmup , AutoModelForSeq2SeqLM
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.amp.autocast_mode import autocast 
import time
from sklearn.utils.class_weight import compute_class_weight
from torch.amp.grad_scaler import GradScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, balanced_accuracy_score,precision_recall_curve, average_precision_score
from sklearn.feature_extraction.text import CountVectorizer
import math
from wordcloud import WordCloud, STOPWORDS
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm
import os


print('Libraries imported successfully!')


# --- Project plan: ----

# 1. EDA (Understand the data, clean target variables, merge or rename categories) 
# 2. Visualization using bar plots for the labels
# 3. Imbalance handling (we can make 2 models, one with weighted loss only and one with data agumentation + weighted loss)
# 4. Use pre-trained trasformers like ClinicalBERT (hugging face)
#
#

#   This seems like a Multiclass problem rather than multi-label problem 
  

# I should use functions to make things more clear and organized. 
# WILLIAM USE FUNCTIONS! 

#%% Data loading: 


try:
    raw_data = pd.read_csv('mtsamples.csv')
except FileNotFoundError:
    print(f"Error: data file not found.")
        
    # Print basic info
print("\n--- Initial Data Info ---")
raw_data.info()


#%% EDA: 
   
def plot_label_distribution(raw_data, column, title, filename):
    
    """
    Helper function to create and save a bar plot of a column's distribution.
    
    Inputs:
            raw_data = (list) data to analyze 
            column   = (string) column to analyze
            title    = (string) title of the plot
            filename = (string)

    """
    print(f"Generating plot: {title}...")
    plt.figure(figsize=(10, 12))
    
    # Plotting
    sns.countplot(
        y=column,
        data=raw_data,
        order=raw_data[column].value_counts().index,
        palette='viridis'
    )
    
    # Formatting
    plt.title(title, fontsize=16)
    plt.xlabel('Number of Samples', fontsize=12)
    plt.ylabel('Medical Specialty', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show


def plot_top_ngrams(text_data,n,top_k,title):
    """    
    This function creates and saves a bar plot 
    for the top N-grams : 

    Inputs: 
            text_data = the actual data 
            n = (scalar) number of grams to consider
            top_k = (scalar) how many elements to show
            title = (string) title of the plot
    Output: 
            None (just the plot)

    """
    print(f"Generazione plot: {title}...")

    # 1. Set up the Count-Vectorizer:
    # Removing the "stop words" and look for n-grams with (n x n) dimension
    vec = CountVectorizer(
         stop_words='english', 
         ngram_range=(n, n)
     )
    # 2. Fit and data transform 
    X = vec.fit_transform(text_data) 

    # 3. Get the counts 
    # We sum the counts for each n-gram across all documents
    counts = X.sum(axis=0) # Dumb VSC error (struggle to analyze attributes of sparse matrix types)
    # 4. Create a DataFrame:
    # Get the names of the features, the n-grams and the counts
    words = vec.get_feature_names_out()

    # Create a df with n-grams and their counts 
    counts_df = pd.DataFrame({
    'ngram': words,
    'count': np.asarray(counts).flatten() # Conver the row of the matrix into a dense array 
     })
    # 5. Get the top_k
    top_counts_df = counts_df.sort_values(by='count', ascending=False).head(top_k)

     # 6. Plotting
    plt.figure(figsize=(10, 8))
    sns.barplot(
         x='count', 
         y='ngram', 
        data=top_counts_df,
        palette='viridis_r' # Inverted Palette
         )
    plt.title(title, fontsize=16)
    plt.xlabel('Frequenza', fontsize=12)
    plt.ylabel('N-Gram', fontsize=12)
    plt.tight_layout()
    plt.show()


# Let's create a function to measure the average lenght of the sentece:
def plot_lenghts(text_data):
    """
    This function retrives and plots the average lenght of the sentences.

    Inputs: 
            text_data = (list) data to analyze

    Outputs: 
            None (just the plot)

    """
    sentence_lengths = [len(sentence.split()) for sentence in text_data]

    plt.figure(figsize=(10, 6))
    sns.histplot(sentence_lengths, bins=50, kde=True, color='skyblue')
    plt.title('Distribution of Sentence Lengths')
    plt.xlabel('Sentence Length (words)')
    plt.ylabel('Frequency')
    plt.show()

    print(f"Average sentence length: {np.mean(sentence_lengths):.2f} words")
    print(f"Median sentence length: {np.median(sentence_lengths)} words")
    print(f"Max sentence length: {np.max(sentence_lengths)} words")
    print(f"Min sentence length: {np.min(sentence_lengths)} words")



def plot_length_per_class(data, text_col, class_col, num_classes=6):
    """
    Plots a boxplot showing the distribution of transcription lengths
    for the top N most frequent classes.

    Inputs:
            data = (dataframe) data to analyze
            text_col = (string) name of the text column
            class_col = (string) name of the class column
            num_classes = (scalar) Number of classes to show

    Outputs:
            None (just the plot)

    """
    print(f"Generating plot: Text Length per Class (Top {num_classes})...")
    data_clean = data.dropna(subset=[text_col, class_col])
    data_clean['length'] = data_clean[text_col].str.split().str.len()
    
    # Get top N classes
    top_classes = data_clean[class_col].value_counts().nlargest(num_classes).index
    data_top = data_clean[data_clean[class_col].isin(top_classes)]
    
    plt.figure(figsize=(15, 8))
    sns.boxplot(x=class_col, y='length', data=data_top, order=top_classes, palette='viridis', hue=class_col, legend=False)
    plt.title(f'Transcription Length Distribution for Top {num_classes} Specialties', fontsize=16)
    plt.xlabel('Medical Specialty', fontsize=12)
    plt.ylabel('Length (Number of Words)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('length_per_class.png')
    print("Saved plot to length_per_class.png")
    plt.close()

def plot_ngrams_per_class(data, text_col, class_col, top_k=15, n=1, num_classes=6):
    """
    Plots a grid of bar charts showing the top N-grams for the
    top N most frequent classes.

    Inputs:
            data = (dataframe) data to analyze
            text_col = (string) name of the text column
            class_col = (string) name of the class column
            top_k =  (scalar) how many grams to show
            n = (scalar) number of grams to consider
            num_classes = (scalar) Number of classes to show

    Outputs:
            None (just the plots)
    """
    print(f"Generating plot: Top {n}-grams per Class (Top {num_classes})...")
    data_clean = data.dropna(subset=[text_col, class_col])
    top_classes = data_clean[class_col].value_counts().nlargest(num_classes).index
    
    ncols = 3
    nrows = math.ceil(num_classes / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 7))
    axes = axes.flatten() # Flatten the grid for easy iteration

    for i, specialty in enumerate(top_classes):
        ax = axes[i]
        specialty_text = data_clean[data_clean[class_col] == specialty][text_col]
        
        try:
            vec = CountVectorizer(stop_words='english', ngram_range=(n, n))
            X = vec.fit_transform(specialty_text)
            counts = X.sum(axis=0)      # Compiler gives error but works anyway
            words = vec.get_feature_names_out()
            
            counts_df = pd.DataFrame({'ngram': words, 'count': np.asarray(counts).flatten()})
            top_counts_df = counts_df.sort_values(by='count', ascending=False).head(top_k)
            
            sns.barplot(x='count', y='ngram', data=top_counts_df, ax=ax, palette='viridis_r', hue='ngram', legend=False)
            ax.set_title(f"Top {n}-grams for '{specialty}'", fontsize=14)
            ax.set_xlabel('Frequency')
            ax.set_ylabel(f'{n}-gram')
        except ValueError:
            # This can happen if a specialty has no text after stopword removal
            ax.set_title(f"Could not generate {n}-grams for '{specialty}'", fontsize=14)
            ax.text(0.5, 0.5, 'Not enough data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f'top_{n}grams_per_class.png')
    print(f"Saved plot to top_{n}grams_per_class.png")
    plt.close()

def plot_wordclouds_per_class(data, text_col, class_col, num_classes=6):
    """
    Plots a grid of word clouds for the top N most frequent classes.

    Inputs:
            data = (dataframe) data to analyze
            text_col = (string) name of the text column
            class_col = (string) name of the class column
            num_classes = (scalar) Number of classes to show

    Outputs:
            None (just the plots)

    """
    print(f"Generating plot: Word Clouds per Class (Top {num_classes})...")
    data_clean = data.dropna(subset=[text_col, class_col])
    top_classes = data_clean[class_col].value_counts().nlargest(num_classes).index
    
    ncols = 3
    nrows = math.ceil(num_classes / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 6))
    axes = axes.flatten()

    custom_stopwords = set(STOPWORDS)
    
    wordcloud = WordCloud(stopwords = custom_stopwords, 
                          background_color='white', 
                          width=800, 
                          height=400, 
                          colormap='viridis',
                          max_words=100)

    for i, specialty in enumerate(top_classes):
        ax = axes[i]
        specialty_text = " ".join(data_clean[data_clean[class_col] == specialty][text_col])
        
        if specialty_text:
            wc = wordcloud.generate(specialty_text)
            ax.imshow(wc, interpolation='bilinear')
            ax.set_title(specialty, fontsize=16)
        else:
            ax.set_title(f"No text for '{specialty}'", fontsize=16)
        
        ax.axis('off')
        
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig('wordclouds_per_class.png')
    print("Saved plot to wordclouds_per_class.png")
    plt.close()

def get_bert_embeddings(texts, model, tokenizer, device, batch_size=32):
    """
    Passes a list of texts through a BERT model and returns their
    [CLS] token embeddings.
    """
    model.eval()
    all_embeddings = []
    
    print(f"Extracting embeddings in batches of {batch_size}...")
    # Use tqdm for a progress bar if you have it: from tqdm import tqdm
    # for i in tqdm(range(0, len(texts), batch_size)):
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            # Get the model's last hidden state
            outputs = model(**inputs)
        
        # Get the [CLS] token embedding (the first token, index 0)
        # This 768-dim vector represents the "summary" of the sequence
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)
        
    return np.concatenate(all_embeddings, axis=0)

# REPLACEMENT for your old plot_tsne function
def plot_tsne_bert(data, text_col, class_col):
    """
    Plots a t-SNE 2D visualization using ClinicalBERT embeddings,
    colored by class. This is computationally expensive!
    """
    print(f"Generating plot: t-SNE visualization (from BERT embeddings)...")
    
    # 1. Load the BASE model (no classification head)
    # We do this here to keep the function self-contained
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Use AutoModel, NOT AutoModelForSequenceClassification
        model = AutoModel.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Error loading model for t-SNE: {e}")
        return

    # 2. Sample the data (t-SNE on 3000+ samples is very slow)
    # Let's take a stratified sample to speed things up
    n_samples_per_class = 75
    sample_data = data.groupby(class_col, group_keys=False).apply(
        lambda x: x.sample(min(len(x), n_samples_per_class), random_state=42)
    )
    print(f"Running t-SNE on a stratified sample of {len(sample_data)} texts...")
    
    # 3. Get Embeddings
    texts = sample_data[text_col].tolist()
    embeddings = get_bert_embeddings(texts, model, tokenizer, device, batch_size=16)
    
    # 4. Apply t-SNE
    print("Running t-SNE calculation (this may take a few minutes)...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,  # Standard value
        random_state=42,
        n_jobs=-1       # Use all cores
    )
    tsne_results = tsne.fit_transform(embeddings)
    
    # 5. Add results to DataFrame
    sample_data['tsne-1'] = tsne_results[:, 0]
    sample_data['tsne-2'] = tsne_results[:, 1]
    
    # 6. Plot
    plt.figure(figsize=(16, 12))
    sns.scatterplot(
        x='tsne-1', y='tsne-2',
        hue=class_col,
        palette=sns.color_palette("viridis", n_colors=sample_data[class_col].nunique()),
        data=sample_data,
        legend="full",
        alpha=0.8,
        s=50
    )
    plt.title('t-SNE Visualization of *ClinicalBERT Embeddings* by Specialty', fontsize=18)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('tsne_visualization_bert.png', bbox_inches='tight')
    print("Saved plot to tsne_visualization_bert.png")
    plt.close()
    
    # 7. Clean up GPU memory!
    # This is important before you start training your main model
    del model
    del tokenizer
    torch.cuda.empty_cache()
    print("t-SNE model and cache cleared from GPU memory.")


# We plot the 40 medical specialty classes: 

plot_label_distribution(raw_data, 'medical_specialty', 'Medical Specialty Distribution', 'medical_specialty_distribution.png')


# EDA Visualization: 

# We analyze a "Transcription" columm removing NaN:
# We make sure to not pass NaN:

# Global lenght and N-Grams

try: 
    text_data = raw_data.dropna(subset=['transcription'])['transcription']

    # Plot the average lenght: 
    print("\n--- Plotting Sentence Lengths ---")
    plot_lenghts(text_data=text_data)

    # Plot for N = 1
    plot_top_ngrams(text_data,n=1,top_k=20,title='Top 20 Most Frequently Occuring Unigrams')

    # Plot for N = 2
    plot_top_ngrams(text_data,n=2,top_k=20,title='Top 20 Most Frequently Occuring Bigrams')

    # Plot for N = 3
    plot_top_ngrams(text_data,n=3,top_k=20,title='Top 20 Most Frequently Occuring Trigrams')
except KeyError:
    print('"transcription" column not found.')
except Exception as e:
    print(f"An error occurred: {e}")


# Class specific lenght and N-Grams

try:
    # This data is used for class-specific plots
    # We drop NaNs from both columns needed
    eda_data = raw_data.dropna(subset=['transcription', 'medical_specialty'])

    # Call new plots
    plot_length_per_class(eda_data, 'transcription', 'medical_specialty', num_classes=6)
    plot_ngrams_per_class(eda_data, 'transcription', 'medical_specialty', n=1, num_classes=6)
    plot_wordclouds_per_class(eda_data, 'transcription', 'medical_specialty', num_classes=6)

except Exception as e:
    print(f"An error occurred during class-specific EDA: {e}")


# Some Notes: 

# There are some classes with very few samples (ex. Hospice or Allergy)
# For now, let's keep the labels as they are and not 

# We have two very different problems, and we need to separate them:
# Problem A: Small, Real Classes (e.t., Allergy / Immunology with 7 samples, Hospice with 6).
# Problem B: Vague, Noisy Classes (e.g., Consult - History and Phy. with 516 samples, SOAP / Chart with 166).

# Problem B can be seen as "garbage in, garbage out problem"


# I thinks the core idea is to build a first model with dropped labels (around -20) and then a second model specifically trained to deal with the biggest class "surgery". 
# Then idk if we can merge the two models into a third one (best of both worlds model) or just testing the two models. 


# Idea and mindmap: 

# Model 1 = The baseline, this model uses the original data (untuched). It will likely perform bad. 
# Model 2 = The improvement, this model is the one i just coded, it will use a cleaned version of the dataset. 
# Model 3 = The improvement on steroids, we'll perform a selective oversampling of the minority classes and train the model using the clean and agumented data.
# 
# 
# We can also train smaller models with different architectures (not transformers)
# To underline that they are indeed the best choice.  
# 
#
# 
# For model 3, we aren't trying to add bias; we're trying to fix the existing bias. The original dataset is heavily biased towards 'Surgery'. 
# We want to counteract that bias by giving the minority classes a stronger voice 
# How do we do this? We use "back translation"

# There is also the idea of data augmentatio using GAN for data augmentation but i'm not sure. 

# How back translation works? 
# It's a simple, meaning-preserving way to paraphrase a sentence. 
# Original (EN) -> Translate (German) -> Translate Back (EN)


# The pipeline is the following for model 3: 

# Isolate Minority Classes: 
# Identify the classes we want to augment (e.g., all classes with fewer than 100 samples in the train_df).

# Augment Only the Training Set: Never, ever augment the validation or test sets. 
# We must validate on real, unseen data.

# Apply Back Translation:

    # Take the transcription text for the minority classes in train_df.
    # Run them through a back-translation pipeline (e.g., en-de-en or en-it-en).  
    # We can even do it 2-3 times with different languages (e.g., French, German, Spanish) to get multiple variations for each original sample (As Francesco suggested). 
    # Create the New Training Set: Combine the original train_df with the new, augmented-minority-class samples.
    # Re-Train: Train the ClinicalBERT model (Model 2) on this new, larger, and more balanced training set. We should still use the weighted loss function, as the data will likely still be imbalanced, just less severely.


#%%  Data Cleaning and Analysis: 

def clean_medical_specialties(data, min_samples=20):
    """
    Cleans the 'medical_specialty' column by removing noisy labels and 
    filtering out specialties with too few samples.

    Inputs: 
            "data" = data to clean 
            "min_samples" = threshold for the minimum amount of samples in each label (20 by default)

    Output: 
            "data" = cleaned data
    """
    print(f"Original dataset shape: {data.shape}")
    
    # 1. Drop rows with missing transcriptions or specialties
    data = data.dropna(subset=['transcription', 'medical_specialty'])
    print(f"Shape after dropping NaNs: {data.shape}")

    # Strip leading/trailing whitespace from the column to ensure clean matching.
    data['medical_specialty'] = data['medical_specialty'].str.strip()

    # 2. Define and remove "noisy" labels (document types, not specialties)
    noisy_labels = [
        'SOAP / Chart / Progress Notes',
        'Consult - History and Phy.',
        'Office Notes',
        'Letters',
        'Discharge Summary',
        'Emergency Room Reports',
        'Radiology',  # Often a report type, not a specialty itself
        'General Medicine' # Too broad, often overlaps with others
    ]
    
    # Let's see how many samples these noisy labels represent
    noisy_count = data['medical_specialty'].isin(noisy_labels).sum()
    print(f"Found {noisy_count} samples with noisy labels to remove.")
    
    data = data[~data['medical_specialty'].isin(noisy_labels)]
    print(f"Shape after removing noisy labels: {data.shape}")

    # 3. Handle class imbalance by removing specialties with < min_samples
    specialty_counts = data['medical_specialty'].value_counts()
    specialties_to_keep = specialty_counts[specialty_counts >= min_samples].index
    
    original_classes = data['medical_specialty'].nunique()
    print(f"Original number of classes: {original_classes}")
    
    data = data[data['medical_specialty'].isin(specialties_to_keep)]
    print(f"Shape after removing classes with < {min_samples} samples: {data.shape}")
    
    final_classes = data['medical_specialty'].nunique()
    print(f"Final number of classes: {final_classes}")
    
    return data.reset_index(drop=True)

def preprocess_data(data):
    """
    Main preprocessing function to load, clean, and analyze the data. 
    The "clean_medical_specialties" is inside this function.

    Input: data = raw data
    Output: cleaned_data = cleaned data 

    """
    # Drop the 'Unnamed: 0' column which is just an old index
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    
    # Clean the labels
    cleaned_data = clean_medical_specialties(data, min_samples=30)
    
    # Plot the *new* distribution
    plot_label_distribution(cleaned_data, 'medical_specialty', 
                      'Cleaned Medical Specialty Distribution', 
                      'cleaned_medical_specialty_distribution.png')
    
    return cleaned_data


print("\n--- Starting Data Cleaning & Analysis ---")

    
cleaned_data = preprocess_data(raw_data)

print("\n--- Cleaned Data Info ---")
cleaned_data.info()
print("\n--- New Class Distribution ---")
print(cleaned_data['medical_specialty'].value_counts())

#
# We can plot the tsne: 
plot_tsne_bert(cleaned_data, 'transcription', 'medical_specialty')
# We need this for our model!
NUM_LABELS = cleaned_data['medical_specialty'].nunique()
print(f"\nTotal number of classes for model: {NUM_LABELS}")


#%% Model loading (Clinical BERT)

# Which model to use ? We need a pre-trained Trasformer (NLP) capable of multi-class classification with medical text. 
# The best one available is Clinical BERT  


# We can also explore other models like "tinyBERT" which requires less computation. 
# We should also demostrate why we use transformers by showing that "classic" architectures fails badly with NLP classification. 

# num_labels = NUM_LABELS 

# 1. Load Tokenizer
# This tokenizer is specifically trained on clinical notes (MIMIC-III)
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 2. Load Model
# We use 'AutoModelForSequenceClassification'
# This automatically adds a classification layer (the "head") on top of the base ClinicalBERT model.
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=NUM_LABELS  
)

# Check if your GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on: {device}")

# tokenizing a single transcription
sample_text = cleaned_data['transcription'].iloc[0]
inputs = tokenizer(sample_text, 
                   return_tensors="pt",  # Return PyTorch tensors
                   max_length=512,       # BERT's maximum
                   truncation=True,      # Truncate long notes
                   padding="max_length") # Pad shorter notes to 512

print("\nTokenized input shape:", inputs['input_ids'].shape)
inputs = inputs.to(device) # Move data to the GPU


#%% Focal Loss function class: 

class FocalLoss(nn.Module):
    """
    Custom Focal Loss module.
    
    This combines the class-balancing 'alpha' (the balance_weights_tensor) 
    and the easy-example-focusing 'gamma'.

    The idea behind this is to exploit the cross entrophy: 
        
        FL (pt) =  - alpha * (1 - pt)^gamma * log(pt)
    
        alpha = class weight
        gamma = focusing parameter
    
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (torch.Tensor, optional): A tensor of weights for each class. 
                                            This is the 'balance_weights_tensor'.
                                            Shape: (num_classes,)
            gamma (float, optional): The focusing parameter. Defaults to 2.0.
            reduction (str, optional): 'mean', 'sum', or 'none'. Defaults to 'mean'.
        """
        super(FocalLoss, self).__init__()
        # We need to register alpha as a buffer so it moves to the .to(device)
        # along with the model.
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None # Handle the case where no weights are given
            
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        Forward pass.
        
        Args:
            logits (torch.Tensor): Raw model output (logits).
                                   Shape: (batch_size, num_classes)
            labels (torch.Tensor): Ground truth labels.
                                   Shape: (batch_size,)
        """
        
        # 1. Calculate the log-probabilities (log_pt) for the *correct* class
        #    log_softmax is numerically stable
        log_softmax = F.log_softmax(logits, dim=1)
        
        #    Use .gather() to pick the log_softmax value for the correct label
        #    labels.view(-1, 1) changes shape from [batch_size] to [batch_size, 1]
        log_pt = log_softmax.gather(1, labels.view(-1, 1))
        
        # 2. Calculate the probability (pt)
        #    pt = exp(log_pt)
        pt = log_pt.exp()

        # 3. Calculate the main focal loss term
        #    This is: (1 - pt)^gamma * NLLLoss
        #    NLLLoss is just -log_pt
        #    So, term = (1 - pt)^gamma * (-log_pt)
        focal_term = (1 - pt)**self.gamma
        loss = focal_term * (-log_pt)

        # 4. Apply the (optional) alpha class weights
        if self.alpha is not None:
            # self.alpha is shape [num_classes]
            # We use .gather() to pick the alpha weight for the correct label
            alpha_t = self.alpha.gather(0, labels)
            
            # Reshape alpha_t from [batch_size] to [batch_size, 1] to match 'loss'
            loss = alpha_t.view(-1, 1) * loss

        # 5. Apply the final reduction (mean, sum, or none)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


#%% Data partitioning and Tokenization: 

# 1. Create Label Mappings
# The model needs integer labels, not strings.
# We create two dictionaries to map between them.

# We should add some plot to visualize how the tokenization happened?

#Get unique specialties:
unique_specialties = sorted(cleaned_data['medical_specialty'].unique())


# Create mappings
# label2id: {'Cardiovascular / Pulmonary': 0, 'Neurology': 1, ...}
label2id = {label: i for i, label in enumerate(unique_specialties)}
# id2label: {0: 'Cardiovascular / Pulmonary', 1: 'Neurology', ...}
id2label = {i: label for i, label in enumerate(unique_specialties)}

# We must update the model's config to know about these labels
model.config.label2id = label2id
model.config.id2label = id2label

# 2. Add integer 'label' column to our DataFrame
cleaned_data['label'] = cleaned_data['medical_specialty'].map(label2id)


# 3. Split the data
# We split into 80% train, 20% validation.
# stratify=cleaned_data['label'] ensures both sets have a similar class distribution.
train_df, val_df = train_test_split(
    cleaned_data,
    test_size=0.2,
    stratify=cleaned_data['label'],
    random_state=42
)

print(f"Training data shape: {train_df.shape}")
print(f"Validation data shape: {val_df.shape}")


# Reset index just to be clean
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# We calculate class weights to implement the weighted loss function 

# Strategy 1: 'balanced' (original model)
print("Calculating 'balanced' class weights...")
train_labels = train_df['label'].values
unique_labels = np.unique(train_labels)

class_weights_balanced = compute_class_weight(
    'balanced', 
    classes=unique_labels, 
    y=train_labels
)
balanced_weights_tensor = torch.tensor(class_weights_balanced, dtype=torch.float).to(device)


# Strategy 2: 'log_smoothed' (The new method)
print("Calculating 'log_smoothed' class weights...")
counts = train_df['label'].value_counts().sort_index()

# Use log(count + 1) to smooth. Add epsilon for stability if a class had 0.
log_weights = 1.0 / np.log(counts.values + 1e-6)

# Normalize weights so their sum is roughly the number of classes
log_weights = (log_weights / np.sum(log_weights)) * NUM_LABELS

log_smoothed_weights_tensor = torch.tensor(log_weights, dtype=torch.float).to(device)


# This dictionary will be looped over.
# We also add 'unweighted' to test pure Focal Loss (alpha=None)
weight_strategies = {
    "balanced": balanced_weights_tensor,
    "log_smoothed": log_smoothed_weights_tensor,
    "unweighted": None  # This will pass alpha=None to FocalLoss
}

print(f"\nCreated {len(weight_strategies)} weighting strategies to test: {list(weight_strategies.keys())}")


#%% Create PyTorch Dataset Class

# Now we create a custom Dataset class.
# This class will handle tokenizing the text on-the-fly.
# This is the standard way to feed data to a model in PyTorch.

class MedicalTranscriptionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        # We get the texts and labels from the dataframe
        self.transcriptions = self.dataframe['transcription']
        self.labels = self.dataframe['label']

    def __len__(self):
        # This returns the total number of samples
        return len(self.dataframe)

    def __getitem__(self, idx):
        # This gets a single sample (transcription) at index 'idx'
        text = self.transcriptions[idx]
        
        # Get the integer label
        label = self.labels[idx]

        # Tokenize the text
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length", # Always pad to max_length
            return_tensors="pt"   # Return PyTorch tensors
        )

        # The tokenizer returns a dictionary. We need to "squeeze"
        # the tensors to remove the batch dimension (which is 1)
        # The Trainer will re-add the batch dimension.
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long) # PyTorch needs 'labels'
        }



#%% Instantiate Datasets

# Now we use the class to create our two dataset objects.
# We already have the 'tokenizer' and the DataFrames 'train_df', 'val_df'

print("\nCreating training dataset...")
train_dataset = MedicalTranscriptionDataset(
    dataframe=train_df,
    tokenizer=tokenizer
)

print("Creating validation dataset...")
val_dataset = MedicalTranscriptionDataset(
    dataframe=val_df,
    tokenizer=tokenizer
)

# Let's check one sample
sample = train_dataset[0]
print("\n--- Sample from Training Dataset ---")
print("Input IDs shape:", sample['input_ids'].shape)
print("Attention Mask shape:", sample['attention_mask'].shape)
print("Label:", sample['labels'])
print("Label (string):", id2label[sample['labels'].item()])

# %%  Training and  Visualization: 

#%% Model Training, Visualization, and Final Evaluation
# This cell now includes the gamma search loop, differential learning rates,
# AND the new weighting strategy comparison.

# Define Compute Metrics Function 
def compute_metrics(eval_pred):
    """
    Computes accuracy and F1 score from logits and labels.
    'eval_pred' is a tuple (logits, labels).
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1) # Get the index of the max logit
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    b_acc = balanced_accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "balanced_accuracy": b_acc,
    }


# Setup DataLoaders 
print("Creating DataLoaders...")
train_loader = DataLoader(
    train_dataset,
    batch_size=4, # per_device_train_batch_size
    shuffle=True, # Shuffle training data
)
val_loader = DataLoader(
    val_dataset,
    batch_size=8, #  per_device_eval_batch_size
)

# Setup Scaler 
scaler = GradScaler(device="cuda")

# 4. Training Configuration 
num_train_epochs = 3 
output_dir = "./clinical_bert_classifier"
accumulation_steps = 4 

# Hyperparameter Search Setup 
gamma_values_to_test = [1.0, 2.0, 3.0] # <-- Reduced for speed 
all_experiment_results = []
all_training_histories = [] 

overall_best_f1 = 0.0
overall_best_model_path = "" 

# Wheighting strategy loop: 
for strategy_name, alpha_tensor in weight_strategies.items():
    
    # --- INNER LOOP FOR GAMMA ---
    for gamma_val in gamma_values_to_test:
        print(f"\n========================================================")
        print(f"--- STARTING EXPERIMENT ---")
        print(f"--- STRATEGY: {strategy_name}")
        print(f"--- GAMMA:    {gamma_val}")
        print(f"========================================================")

        # Re-initialize model for a fresh run
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=model.config # Re-use config with label mappings
        ).to(device)
        
        # Setup Optimizer with Differential Learning Rates
        lr_head = 1e-4 
        lr_body = 2e-5 

        head_param_names = [name for name, param in model.named_parameters() if name.startswith('classifier')]
        
        params_body = [
            param for name, param in model.named_parameters()
            if name not in head_param_names and param.requires_grad
        ]
        params_head = [
            param for name, param in model.named_parameters()
            if name in head_param_names and param.requires_grad
        ]

        optimizer_grouped_parameters = [
            {'params': params_body, 'lr': lr_body},
            {'params': params_head, 'lr': lr_head}
        ]

        optimizer = optim.AdamW(optimizer_grouped_parameters)

        print(f"Optimizer created with {len(params_body)} param groups for body (LR={lr_body}) "
              f"and {len(params_head)} for head (LR={lr_head}).")

        # Create Loss Function using loop variables
        # alpha_tensor comes from our new outer loop
        # gamma_val comes from our inner loop
        loss_fct = FocalLoss(alpha=alpha_tensor, gamma=gamma_val)
        print(f"FocalLoss created with strategy='{strategy_name}' and gamma={gamma_val}")
        
        # Make best_model_path unique for this run
        best_model_path = f"{output_dir}/best_model_strategy_{strategy_name}_gamma_{gamma_val}"

        # --- Scheduler Setup 
        num_training_steps = (len(train_loader) // accumulation_steps) * num_train_epochs
        num_warmup_steps = int(num_training_steps * 0.1) 

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        print(f"Scheduler initialized. Total steps: {num_training_steps}, Warmup: {num_warmup_steps}")

        # --- Training State 
        best_f1 = 0.0 
        training_history = [] 

        print(f"Starting training on {device} for {num_train_epochs} epochs...")

        for epoch in range(num_train_epochs):
            start_time = time.time()

            # --- Training Phase ---
            model.train() 
            total_train_loss = 0
            optimizer.zero_grad()

            for i, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                with autocast(device_type='cuda'):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits
                    loss = loss_fct(logits, labels)
                    loss = loss / accumulation_steps 

                scaler.scale(loss).backward()
                total_train_loss += loss.item() * accumulation_steps

                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step() 
                    optimizer.zero_grad()

                    if (i + 1) % (50 * accumulation_steps) == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        print(
                            f"  S:{strategy_name} G:{gamma_val} | "
                            f"Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader)}, "
                            f"Avg Loss: {total_train_loss / (i + 1):.4f}, LR: {current_lr:.8f}"
                        )

            avg_train_loss = total_train_loss / len(train_loader)

            # --- Validation Phase ---
            model.eval()
            all_logits = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        logits = outputs.logits
                    all_logits.append(logits.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

            eval_pred = (np.concatenate(all_logits), np.concatenate(all_labels))
            metrics = compute_metrics(eval_pred)
            epoch_f1 = metrics['f1_weighted']
            epoch_acc = metrics['accuracy']
            epoch_b_acc = metrics['balanced_accuracy']

            end_time = time.time()
            print(f"\n--- Epoch {epoch + 1}/{num_train_epochs} Complete (Strategy={strategy_name}, Gamma={gamma_val}) ---")
            print(f"Time: {end_time - start_time:.2f}s")
            print(f"Avg Train Loss: {avg_train_loss:.4f}")
            print(f"Validation F1 (Weighted): {epoch_f1:.4f}")
            print(f"Validation Accuracy: {epoch_acc:.4f}")
            print(f"Validation Balanced Accuracy: {epoch_b_acc:.4f}")

            # Save metrics to history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_f1': epoch_f1,
                'val_accuracy': epoch_acc,
                'val_balanced_accuracy': epoch_b_acc
            })

            # Save Best Model *for this run*
            if epoch_f1 > best_f1:
                best_f1 = epoch_f1
                print(f"New best model for {strategy_name}/gamma={gamma_val}! F1: {best_f1:.4f}. Saving to '{best_model_path}'...")
                model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)

            print("----------------------------------\n")

        print(f"--- Training Complete for Strategy={strategy_name}, Gamma={gamma_val} ---")
        print(f"Best Validation F1 score achieved: {best_f1:.4f}")

        # Add experiment params to this run's history
        for record in training_history:
            record['gamma'] = gamma_val
            record['strategy'] = strategy_name # 
        
        all_training_histories.extend(training_history)

        # Save results for this experiment 
        all_experiment_results.append({
            'strategy': strategy_name, # ì
            'gamma': gamma_val,
            'best_f1': best_f1,
            'best_model_path': best_model_path
        })
        
        # Track the OVERALL best model 
        if best_f1 > overall_best_f1:
            overall_best_f1 = best_f1
            overall_best_model_path = best_model_path
            print(f"This is the NEW OVERALL BEST model. Path: {overall_best_model_path} !!!")


# Final Experiment Summary 
print("\n==============================================")
print("--- All Experiments Complete ---")
results_df = pd.DataFrame(all_experiment_results)
# Sort by F1 score to see the best runs at the top
print(results_df.sort_values(by='best_f1', ascending=False))

# Master history df:
all_history_df = pd.DataFrame(all_training_histories)

print("\n--- Overall Best Model ---")
print(f"Path: {overall_best_model_path}")
print(f"Best F1: {overall_best_f1:.4f}")
print("==============================================")

#%%

# -----------------------------------------------------------------
# 5. PLOTTING AND FINAL EVALUATION 
# -----------------------------------------------------------------


print("\n--- Generating Plots and Final Report ---")

# --- Plot 1: Training & Validation Metrics (FOR THE OVERALL BEST RUN) ---
print("Generating plot for the *single best* run...")
try:
    # Find the best run from our results
    best_run = results_df.loc[results_df['best_f1'].idxmax()]
    best_gamma_val = best_run['gamma']
    best_strategy_name = best_run['strategy']
    
    print(f"Best model was: Strategy='{best_strategy_name}', Gamma={best_gamma_val} (F1: {best_run['best_f1']:.4f})")

    # Filter the master history for just this one run
    best_history_df = all_history_df[
        (all_history_df['gamma'] == best_gamma_val) & 
        (all_history_df['strategy'] == best_strategy_name)
    ]
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(best_history_df['epoch'], best_history_df['train_loss'], 
             label=f'Train Loss (Best Run)', marker='o')
    plt.title(f'Best Run Loss (Strategy: {best_strategy_name}, Gamma: {best_gamma_val})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot F1, Accuracy, and Balanced Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(best_history_df['epoch'], best_history_df['val_f1'], 
             label=f'Validation F1', marker='o')
    plt.plot(best_history_df['epoch'], best_history_df['val_accuracy'], 
             label=f'Validation Accuracy', marker='s', linestyle='--')
    plt.plot(best_history_df['epoch'], best_history_df['val_balanced_accuracy'], 
             label=f'Validation Balanced Acc', marker='^', linestyle=':')
    plt.title(f'Best Run Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics_BEST_run.png')
    print("Saved best run training metrics to 'training_metrics_BEST_run.png'")
    plt.close()

except Exception as e:
    print(f"Error generating best run training plot: {e}")


# --- Plot 2: Confusion Matrix & Classification Report (from OVERALL best model) ---

# We use the 'overall_best_model_path' variable we saved from the loop
print(f"Loading OVERALL best model from '{overall_best_model_path}' for final evaluation...")
try:
    # Load the best model we saved during training
    best_model = AutoModelForSequenceClassification.from_pretrained(overall_best_model_path)
    best_model.to(device)
    best_model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = best_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Get final predictions
    final_logits = np.concatenate(all_logits)
    final_labels = np.concatenate(all_labels)
    final_preds = np.argmax(final_logits, axis=1)
    
    label_names = [id2label[i] for i in range(len(id2label))]
    
    # 1. Print Classification Report
    print("\n--- Final Classification Report (Best Model) ---")
    report = classification_report(final_labels, final_preds, target_names=label_names)
    print(report)
    
    # 2. Generate and Save Confusion Matrix
    cm = confusion_matrix(final_labels, final_preds)
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, 
                yticklabels=label_names)
    plt.title('Confusion Matrix (Best Model)', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix plot to 'confusion_matrix.png'")
    plt.close()

    # 3. Generate and Save Normalized Confusion Matrix
    print("Generating Normalized Confusion Matrix...")
    cm_normalized = confusion_matrix(final_labels, final_preds, normalize='true')
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues', 
        xticklabels=label_names, 
        yticklabels=label_names
    )
    plt.title('Normalized Confusion Matrix (by True Label)', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_normalized.png')
    print("Saved normalized confusion matrix plot to 'confusion_matrix_normalized.png'")
    plt.close()

except Exception as e:
    print(f"Error during final evaluation: {e}")


# --- Plot 3: Training Dynamics (Strategy & Gamma Comparison) ---
print("Generating Training Dynamics Comparison Plot...")
try:
    # We use sns.relplot to create a grid of plots:
    # One column for each 'strategy', with 'gamma' as the hue inside.
    g = sns.relplot(
        data=all_history_df,
        x='epoch',
        y='val_f1',
        hue='gamma',        # Color lines by gamma value
        style='gamma',      # Use different line styles for gamma
        col='strategy',     # <-- NEW: Create columns for each strategy
        kind='line',        # Make it a line plot
        markers=True,
        palette='viridis',
        lw=2,
        height=5,           # Height of each subplot
        aspect=1.2          # Aspect ratio
    )
    
    g.fig.suptitle('Validation F1-Score Dynamics by Strategy and Gamma', y=1.03, fontsize=16)
    g.set_axis_labels("Epoch", "Validation F1-Score (Weighted)")
    
    plt.tight_layout()
    plt_file_name = 'strategy_gamma_comparison_f1.png'
    plt.savefig(plt_file_name)
    print(f"Saved plot to '{plt_file_name}'")
    plt.close()

except Exception as e:
    print(f"Error generating dynamics comparison plot: {e}")


#%%

print("Generating Training Dynamics (Balanced Accuracy) Comparison Plot...")
try:
    # Repeat the same plot, but for 'val_balanced_accuracy'
    g = sns.relplot(
        data=all_history_df,
        x='epoch',
        y='val_balanced_accuracy',  
        hue='gamma',        
        style='gamma',      
        col='strategy',     
        kind='line',        
        markers=True,
        palette='viridis',
        lw=2,
        height=5,           
        aspect=1.2          
    )
    
    g.fig.suptitle('Validation Balanced Accuracy Dynamics by Strategy and Gamma', y=1.03, fontsize=16)
    g.set_axis_labels("Epoch", "Validation Balanced Accuracy")
    
    plt.tight_layout()
    plt_file_name = 'strategy_gamma_comparison_balanced_accuracy.png'
    plt.savefig(plt_file_name)
    print(f"Saved plot to '{plt_file_name}'")
    plt.close()

except Exception as e:
    print(f"Error generating balanced accuracy dynamics plot: {e}")


#%%
# ... (Your Precision-Recall Curve cell is fine as-is) ...
# ... (It will run on the 'overall_best_model_path' found by the new loop) ...

#%%









# --- Plot 4: Precision-Recall Curve (Best Model) ---
print("Generating Precision-Recall Curve...")
try:

    # We compute ONE VS REST and average the result: 

    # We need:
    # 1. final_labels (true integer labels)
    # 2. final_logits (raw model scores)
    # 3. NUM_LABELS (total number of classes)
    
    # --- 1. Get Probabilities (Scores) ---
    # Apply softmax to logits to get probabilities
    # We use torch.tensor and F.softmax for consistency
    y_scores = F.softmax(torch.tensor(final_logits), dim=1).numpy()

    # --- 2. Binarize True Labels ---
    # Convert labels [0, 2, 1, ..] to a one-hot matrix
    # [[1, 0, 0], [0, 0, 1], [0, 1, 0], ..]
    y_true_bin = label_binarize(final_labels, classes=list(range(NUM_LABELS)))

    # --- 3. Compute PR curve for each class ---
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(NUM_LABELS):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin[:, i], y_scores[:, i]
        )
        average_precision[i] = average_precision_score(
            y_true_bin[:, i], y_scores[:, i]
        )

    # --- 4. Compute Micro-Average ---
    # This aggregates all (true, score) pairs and treats it as one big binary problem
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_scores.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_true_bin, y_scores, average="micro"
    )

    # --- 5. Compute Macro-Average ---
    # This involves interpolating all PR curves to a common recall axis
    
    # First, gather all unique recall points and sort them
    all_recall = np.unique(np.concatenate([recall[i] for i in range(NUM_LABELS)]))
    
    # Then, interpolate each class's precision at these points
    mean_precision = np.zeros_like(all_recall)
    for i in range(NUM_LABELS):
        # np.interp needs x-points (all_recall), 
        # and the original (recall[i], precision[i]) pairs
        # We reverse recall[i] and precision[i] because recall is descending
        mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])

    # Finally, average the interpolated precisions
    mean_precision /= NUM_LABELS

    # The macro-average AP score is the simple average of each class's AP
    average_precision["macro"] = np.mean(list(average_precision.values()))

    # --- 6. Plot the Curves ---
    plt.figure(figsize=(10, 8))
    
    # Plot the Micro-average
    plt.plot(
        recall["micro"],
        precision["micro"],
        label=f'Micro-average PR (AP = {average_precision["micro"]:.3f})',
        color='deeppink',
        linestyle=':',
        lw=3
    )

    # Plot the Macro-average
    plt.plot(
        all_recall,
        mean_precision,
        label=f'Macro-average PR (AP = {average_precision["macro"]:.3f})',
        color='navy',
        linestyle='--',
        lw=3
    )

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Multi-Class Precision-Recall Curve (Best Model)', fontsize=16)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png')
    print("Saved plot to 'precision_recall_curve.png'")
    plt.close()

except Exception as e:
    print(f"Error generating Precision-Recall curve: {e}")


# %% Comments: 


# 4/11 Comments on results (OLD TRAINING WITHOUT FOCUS, WEIGHTS ONLY): 

# The model is currently "bias overfitting"
# These large, generic classes ("Surgery," "Consult," "SOAP/Chart") likely contain so much 
# varied language that they overlap with everything. The model can't find a unique signal, 
# so it guesses, and its guesses are all over the place.

# How to fix this? 

# Focal loss is a thing? 
# Differential Learning rates ? 
# Just surrender on model 2 and try data agumentation (model 3)



# 8/11 New comments: 

# We implemented The focal loss function and the differential learning rates 
# but something backfired. 

# The new model is an expert at identifying minority classes. 
# However, it has learned to do this at the extreme expense of the majority classes
# Now surgery has only 0.06 Recall ect... why? 

# High Gamma (Best Model = 3.0): 
# A gamma of 3.0 tells the model to aggressively ignore samples it finds "easy." 
# The 'Surgery' and 'Consult' classes are so large and generic that the model likely classified them as "easy" very early on.
#  It effectively stopped learning from them.

# Class Weights (alpha): We told the model that misclassifying a single 
# 'Ophthalmology' sample is a massive penalty, while misclassifying a
# 'Surgery' sample is a tiny penalty.


# A small comment on the gamma comparison with F1 and Balanced accuracy: 
# The two plots don't agree on the number of epochs, this conflict between 
# the two main metrics is a huge red flag. It's screaming that the model is fighting the data. 
# We are trying to find a single "best" model, but our metrics can't even agree on what "best" is.

# I think we hit the limit of model 2 (loss function alone can't correct an unbalanced dataset) It is time to move forward to model 3: 


# 9/11 Comments: 

# I'm going to change the way the alpha weights are calculated. 
# I will change from balance only to "balance" and "log-smoothing" 
# This way we can compare the two. 


# 10/11 Comments: 

# I performed the training, it takes roughly one hour and ten minutes. 
# The results, as expected, are very bad due to unbalanced data. 
# According to the F1 score the Best model was: Strategy='log_smoothed', Gamma=3.0 (F1: 0.3643)



# Let's perform the data agumentation; How? 

# In medical text, you cannot afford to randomly swap, delete, or insert words
# Since we use a transformer, we don't need to pay for Google's API. 
# We can run state-of-the-art translation models locally on the GPU for free. 

# Model to use: 
# The best models for this are from the Helsinki-NLP opus-mt series. 
# They are lightweight, fast, and high-quality (they say).

# We would need two models ( one from ENG to GER and the other from GER to ENG)

# A great "starter pack" for augmentation:    
# German (de): A high-quality Germanic language model.
# French (fr) or Italian (it): A high-quality Romance language model.
# Russian (ru): A high-quality Slavic language model.

# Are there alternatives ? 

# 1. Masked Language Model (MLM) Augmentation 
# How ? 
# Generates in-domain variations. 
# Libraries like nlpaug can help implement this. 
# Slightly riskier than back-translation, as it might predict a word that changes 
# the meaning, but it's generally safe if you only mask one or two words

# 2. Simple Synonym Replacement (WordNet) or  Easy Data Augmentation (EDA) (we should avoid these as they are risky.)

# AGGRESSIVE CLEANING! 
# Aside from data agumentation, we can remove some labels that only introduce noise in the dataset: 
# Lables too generic "Letters", "SOAP" or "Office Notes" can be removed 
# Labels too similar like "Neurology and Neurosurgery" can be mixed togheter 
# Labels with less than 50 samples can be discarded (i think that even if we perform data augmentation we can't balance those classes)

# The new model (model 3) should be trained on agumented and pre-processed data (the pre-processing function will change considering what i wrote in the lines above)




#%%  Data Augmentation and Aggressive Clening: 

# Cleaning strategy: 

# Merge, drop and keep:

# Surgery (Base Class): Surgery (1088)

   # Merge with: Bariatrics (18) -> This is a type of GI surgery.

   # Merge with: Cosmetic / Plastic Surgery (27) -> This is a type of surgery.

   # New Class: "Surgery" (Total: 1133)

# Neurology (Base Class): Neurology (223)

 #   Merge with: Neurosurgery (94)

  #  New Class: "Neurology / Neurosurgery" (Total: 317)


# Drop all the noisy labels (just as before)


# Consult - History and Phy. (516) - A neurologist, a surgeon, and a cardiologist all write "Consults."

# SOAP / Chart / Progress Notes (166) - Document type.

# Discharge Summary (108) - Document type.

# Emergency Room Reports (75) - Document type / department.

# General Medicine (259) - Too broad. Overlaps with everything.

# Radiology (273) - This is a specialty, but its reports are structured dictations (what they see), not patient transcriptions (what they do). It's linguistically very different and adds noise.

# Office Notes (50) - Document type.

# Letters (23) - Document type.

# IME-QME-Work Comp etc. (16) - Legal document type.

# Lab Medicine - Pathology (8) - Report type, also too small.

# Autopsy (8) - Report type, also too small.

# We now drop labels with < 30 samples (too few only noise and using data agumentation would just add bias)
# We should remain with 13 well defined classes to augment. 

# First i will clean the data and look at the remaining labels 
# Then i will perform data augmentation

def aggressive_clean_data(data, min_samples=30):
    """
    Performs aggressive cleaning of the 'medical_specialty' column
    based on the Model 3 strategy:
    1. Drops NaNs and 'Unnamed: 0' column.
    2. Merges similar classes (e.g., 'Bariatrics' -> 'Surgery').
    3. Removes a specific list of noisy/document-type labels.
    4. Filters out any remaining classes with < min_samples.
    
    Inputs: 
            "data" = (DataFrame) The raw data to clean.
            "min_samples" = (int) Threshold for the minimum samples per class.

    Output: 
            (DataFrame) The aggressively cleaned and filtered data.
    """
    print("--- Starting AGGRESSIVE Data Cleaning ---")
    print(f"Original dataset shape: {data.shape}")
    
    # Make a copy to avoid SettingWithCopyWarning
    data = data.copy()

    # 1. Initial cleanup
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    
    data['medical_specialty'] = data['medical_specialty'].str.strip()
    data = data.dropna(subset=['transcription', 'medical_specialty'])
    print(f"Shape after dropping NaNs: {data.shape}")

    # 2. Merge similar specialties
    merge_map = {
        'Bariatrics': 'Surgery',
        'Cosmetic / Plastic Surgery': 'Surgery',
        'Neurosurgery': 'Neurology / Neurosurgery',
        'Neurology': 'Neurology / Neurosurgery' # Map base class too
    }
    
    data['medical_specialty'] = data['medical_specialty'].replace(merge_map)
    print("Completed merging 'Bariatrics'/'Cosmetic' into 'Surgery' and 'Neurosurgery' into 'Neurology / Neurosurgery'.")

    # 3. Define and remove "noisy" labels (document types, not specialties)
    noisy_labels = [
        'Consult - History and Phy.',
        'SOAP / Chart / Progress Notes',
        'Discharge Summary',
        'Emergency Room Reports',
        'General Medicine',
        'Radiology',
        'Office Notes',
        'Letters',
        'IME-QME-Work Comp etc.',
        'Lab Medicine - Pathology',
        'Autopsy'
    ]
    
    
    noisy_count = data['medical_specialty'].isin(noisy_labels).sum()
    print(f"Found {noisy_count} samples with noisy labels to remove.")
    
    data = data[~data['medical_specialty'].isin(noisy_labels)]
    print(f"Shape after removing noisy labels: {data.shape}")

    # 4. Handle class imbalance by removing specialties with < min_samples
    specialty_counts = data['medical_specialty'].value_counts()
    specialties_to_keep = specialty_counts[specialty_counts >= min_samples].index
    
    original_classes = data['medical_specialty'].nunique()
    print(f"Classes before min_sample filter: {original_classes}")
    
    data = data[data['medical_specialty'].isin(specialties_to_keep)]
    print(f"Shape after removing classes with < {min_samples} samples: {data.shape}")
    
    final_classes = data['medical_specialty'].nunique()
    print(f"--- Aggressive Cleaning Complete ---")
    print(f"Final number of classes: {final_classes}")
    
    return data.reset_index(drop=True)


try:
     clean_data_m3 = aggressive_clean_data(raw_data, min_samples=50)
     print('Aggressive data cleaning completed successfully.')
except KeyError:
    print('"transcription" or "medical_specialty" column not found.')
except Exception as e:
    print(f"An error occurred: {e}")
   
print("\n--- New Class Distribution (After Aggressive Clean) ---")
print(clean_data_m3['medical_specialty'].value_counts())
print('Mean number of samples per class after cleaning:', clean_data_m3['medical_specialty'].value_counts().mean())
print('Median number of samples per class after cleaning:', clean_data_m3['medical_specialty'].value_counts().median())

# %% EDA wit the new dataframe:

# New labels plot:
# Plot the new distribution

try:
    plot_label_distribution(clean_data_m3, 'medical_specialty', 
                  'Aggressively Cleaned Specialty Distribution', 
                  'cleaned_medical_specialty_m3.png')
except KeyError:
    print('"medical_specialty" column not found.')
except Exception as e:
    print(f"An error occurred: {e}")


NUM_LABELS_M3 = clean_data_m3['medical_specialty'].nunique()
print(f"\nTotal number of classes for Model 3: {NUM_LABELS_M3}")

# One can add the other EDA plots here 

#%% Back translation: 

# Load translation models: 

# ITA, GER
print("Loading translation models...")
models = {}
tokenizers = {}

# We define again "device" if the previous part of the code was not run: 
if 'device' not in globals():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = device



# English to German and back
model_name_en_de = "Helsinki-NLP/opus-mt-en-de"
model_name_de_en = "Helsinki-NLP/opus-mt-de-en"
tokenizers['en_de'] = AutoTokenizer.from_pretrained(model_name_en_de)
models['en_de'] = AutoModelForSeq2SeqLM.from_pretrained(model_name_en_de).to(device)
tokenizers['de_en'] = AutoTokenizer.from_pretrained(model_name_de_en)
models['de_en'] = AutoModelForSeq2SeqLM.from_pretrained(model_name_de_en).to(device)

# English to Italian and back
model_name_en_it = "Helsinki-NLP/opus-mt-en-it"
model_name_it_en = "Helsinki-NLP/opus-mt-it-en"
tokenizers['en_it'] = AutoTokenizer.from_pretrained(model_name_en_it)
models['en_it'] = AutoModelForSeq2SeqLM.from_pretrained(model_name_en_it).to(device)
tokenizers['it_en'] = AutoTokenizer.from_pretrained(model_name_it_en)
models['it_en'] = AutoModelForSeq2SeqLM.from_pretrained(model_name_it_en).to(device)

print("All translation models loaded and moved to device.")

# Back-Translation Function (with Batching) 
def back_translate(texts, lang_pair, batch_size=16):
    """
    Performs back-translation on a list of texts using a specified lang_pair.
    
    Args:
        texts (list): A list of sentences to translate.
        lang_pair (str): 'de' or 'it' to select the models.
        batch_size (int): How many texts to process at once.
        
    Returns:
        list: A list of back-translated sentences.
    """
    
    # Select models for the chosen language pair
    tokenizer_en_xx = tokenizers[f'en_{lang_pair}']
    model_en_xx = models[f'en_{lang_pair}']
    tokenizer_xx_en = tokenizers[f'{lang_pair}_en']
    model_xx_en = models[f'{lang_pair}_en']
    
    back_translated_texts = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # 1. Translate EN -> XX
        inputs_en_xx = tokenizer_en_xx(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        translated_ids_xx = model_en_xx.generate(**inputs_en_xx, max_length=512)
        translated_texts_xx = tokenizer_en_xx.batch_decode(translated_ids_xx, skip_special_tokens=True)

        # 2. Translate XX -> EN
        inputs_xx_en = tokenizer_xx_en(translated_texts_xx, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        translated_ids_en = model_xx_en.generate(**inputs_xx_en, max_length=512)
        back_translated_batch = tokenizer_xx_en.batch_decode(translated_ids_en, skip_special_tokens=True)
        
        back_translated_texts.extend(back_translated_batch)
    
    return back_translated_texts

# Create New Splits & Mappings from Cleaned Data ---

# Create new label mappings
unique_specialties_m3 = sorted(clean_data_m3['medical_specialty'].unique())
label2id_m3 = {label: i for i, label in enumerate(unique_specialties_m3)}
id2label_m3 = {i: label for i, label in enumerate(unique_specialties_m3)}

# Add integer 'label' column
clean_data_m3['label'] = clean_data_m3['medical_specialty'].map(label2id_m3)

# Step split 
# 1. First, split into (train+validation) and (test)
# Hold out 15% of the data for the final test
train_val_df_m3, test_df_m3 = train_test_split(
    clean_data_m3,
    test_size=0.15, # 15% for the final, hold-out test set
    stratify=clean_data_m3['label'],
    random_state=42 
)

# 2. Now, split the (train+validation) set into train and validation
# We'll use 20% of the remaining 85% for validation
train_df_m3, val_df_m3 = train_test_split(
    train_val_df_m3, # We split the 85%, not the full dataset
    test_size=0.2, # 20% of 85% = 17% of total (so ~68% train, 17% val, 15% test)
    stratify=train_val_df_m3['label'], # Stratify on the set we're splitting
    random_state=42
)

print(f"\nCleaned training data shape: {train_df_m3.shape}")
print(f"Cleaned validation data shape: {val_df_m3.shape}")
print(f"Cleaned **TEST** data shape: {test_df_m3.shape} (This is our hold-out set)")

# Reset indices for clean processing
train_df_m3 = train_df_m3.reset_index(drop=True)
val_df_m3 = val_df_m3.reset_index(drop=True)
test_df_m3 = test_df_m3.reset_index(drop=True)

# Identify Augmentation Targets 
print("\n--- Starting Data Augmentation ---")

# Let's set a reasonable target. 'Neurology / Neurosurgery' has ~250.
# Let's augment all classes to have at least 250 samples.

TARGET_SAMPLES = 300
AUGMENTATION_BATCH_SIZE = 16 # Batch size for translation models

class_counts = train_df_m3['label'].value_counts()
augmented_data_list = []

# Loop through each class
for label_id, count in class_counts.items():
    specialty_name = id2label_m3[label_id]
    
    if count < TARGET_SAMPLES:
        n_needed = TARGET_SAMPLES - count
        print(f"Augmenting '{specialty_name}': Need to generate {n_needed} samples.")
        
        # Get all original texts for this class
        minority_texts = train_df_m3[train_df_m3['label'] == label_id]['transcription'].tolist()
        
        n_generated = 0
        while n_generated < n_needed:
            # How many to generate in this batch
            n_to_generate_now = min(AUGMENTATION_BATCH_SIZE, n_needed - n_generated)
            
            # Randomly sample from the original texts (with replacement)
            texts_to_augment = np.random.choice(minority_texts, n_to_generate_now).tolist()
            
            # Alternate between DE and it for variety (as "Wafo suggested" )
            if (n_generated // AUGMENTATION_BATCH_SIZE) % 2 == 0:
                lang_pair = 'de'
            else:
                lang_pair = 'it'
                
            # Perform back-translation
            new_texts = back_translate(texts_to_augment, lang_pair=lang_pair, batch_size=n_to_generate_now)
            
            # Add new (text, label) pairs to our list
            for text in new_texts:
                augmented_data_list.append({
                    'transcription': text,
                    'medical_specialty': specialty_name,
                    'label': label_id
                })
            
            n_generated += n_to_generate_now
            if n_generated % (AUGMENTATION_BATCH_SIZE * 5) == 0:
                print(f"  ... generated {n_generated}/{n_needed} for '{specialty_name}'")

print("Augmentation complete.")

# --- 2.5 Create Final Augmented Training Set ---
augmented_df = pd.DataFrame(augmented_data_list)
train_df_m3_final = pd.concat([train_df_m3, augmented_df]).reset_index(drop=True)

print("\n--- Final Training Set Distribution (After Augmentation) ---")
print(train_df_m3_final['medical_specialty'].value_counts().sort_index())
print(f"\nOld training set size: {len(train_df_m3)}")
print(f"New augmented training set size: {len(train_df_m3_final)}")




# %% Training model 3: 


# 1. Load Model with new configuration
print(f"Loading model for {NUM_LABELS_M3} classes...")
model_m3 = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=NUM_LABELS_M3  
)
# We must update the model's config with the new M3 mappings
model_m3.config.label2id = label2id_m3
model_m3.config.id2label = id2label_m3
model_m3.to(device)
print(f"Model 3 loaded on: {device}")


# 2. Instantiate Datasets
print("\nCreating Model 3 training dataset (with augmentation)...")
train_dataset_m3 = MedicalTranscriptionDataset(
    dataframe=train_df_m3_final,
    tokenizer=tokenizer
)

print("Creating Model 3 validation dataset (clean)...")
val_dataset_m3 = MedicalTranscriptionDataset(
    dataframe=val_df_m3, # The original, non-augmented validation set
    tokenizer=tokenizer
)

print("Creating Model 3 **TEST** dataset (clean, hold-out)...")
test_dataset_m3 = MedicalTranscriptionDataset(
    dataframe=test_df_m3, # Use the new test_df_m3
    tokenizer=tokenizer
)

# 3. Re-calculate Class Weights for the *new* augmented training set
print("Calculating 'balanced' class weights for M3...")
train_labels_m3 = train_df_m3_final['label'].values
unique_labels_m3 = np.unique(train_labels_m3)

class_weights_balanced_m3 = compute_class_weight(
    'balanced', 
    classes=unique_labels_m3, 
    y=train_labels_m3
)
balanced_weights_tensor_m3 = torch.tensor(class_weights_balanced_m3, dtype=torch.float).to(device)

print("Calculating 'log_smoothed' class weights for M3...")
counts_m3 = train_df_m3_final['label'].value_counts().sort_index()
log_weights_m3 = 1.0 / np.log(counts_m3.values + 1e-6)
log_weights_m3 = (log_weights_m3 / np.sum(log_weights_m3)) * NUM_LABELS_M3
log_smoothed_weights_tensor_m3 = torch.tensor(log_weights_m3, dtype=torch.float).to(device)

# This dictionary will be looped over for Model 3
weight_strategies_m3 = {
    "balanced": balanced_weights_tensor_m3,
    "log_smoothed": log_smoothed_weights_tensor_m3,
    "unweighted": None
}
print(f"\nCreated {len(weight_strategies_m3)} weighting strategies for M3: {list(weight_strategies_m3.keys())}")


# 4. Setup DataLoaders 
print("Creating Model 3 DataLoaders...")
train_loader_m3 = DataLoader(
    train_dataset_m3,
    batch_size=4, # per_device_train_batch_size
    shuffle=True, # Shuffle training data
)
val_loader_m3 = DataLoader(
    val_dataset_m3,
    batch_size=8, #  per_device_eval_batch_size
)

print("Creating Model 3 **TEST** DataLoader...")
test_loader_m3 = DataLoader(
    test_dataset_m3,
    batch_size=8, # Use same batch size as validation
    shuffle=False # NEVER shuffle the test set
)

# 5. Setup Scaler (already defined, but for clarity)
scaler_m3 = GradScaler(device="cuda")


# %% --- Step 4: Model 3 Training, Visualization, and Final Evaluation ---

# Configuration 
num_train_epochs = 3 
output_dir_m3 = "./clinical_bert_classifier_m3" # New output directory
accumulation_steps = 4 

# Hyperparameter Search Setup 
gamma_values_to_test = [1.0, 2.0, 3.0] 
all_experiment_results_m3 = []
all_training_histories_m3 = [] 

overall_best_f1_m3 = 0.0
overall_best_model_path_m3 = "" 

# ---OUTER LOOP FOR WEIGHTING STRATEGY ---
for strategy_name, alpha_tensor in weight_strategies_m3.items():
    
    # --- INNER LOOP FOR GAMMA ---
    for gamma_val in gamma_values_to_test:
        print(f"\n========================================================")
        print(f"--- STARTING MODEL 3 EXPERIMENT ---")
        print(f"--- STRATEGY: {strategy_name}")
        print(f"--- GAMMA:    {gamma_val}")
        print(f"========================================================")

        # Re-initialize model for a fresh run, using the M3 config
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=model_m3.config # Use M3 config with correct labels
        ).to(device)
        
        # Setup Optimizer with Differential Learning Rates
        lr_head = 1e-4 
        lr_body = 2e-5 
        head_param_names = [name for name, param in model.named_parameters() if name.startswith('classifier')]
        params_body = [p for n, p in model.named_parameters() if n not in head_param_names and p.requires_grad]
        params_head = [p for n, p in model.named_parameters() if n in head_param_names and p.requires_grad]
        optimizer_grouped_parameters = [
            {'params': params_body, 'lr': lr_body},
            {'params': params_head, 'lr': lr_head}
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters)

        # Create Loss Function using loop variables
        loss_fct = FocalLoss(alpha=alpha_tensor, gamma=gamma_val)
        print(f"FocalLoss created with strategy='{strategy_name}' and gamma={gamma_val}")
        
        # Make best_model_path unique for this run
        best_model_path = f"{output_dir_m3}/best_model_strategy_{strategy_name}_gamma_{gamma_val}"

        # --- Scheduler Setup 
        num_training_steps = (len(train_loader_m3) // accumulation_steps) * num_train_epochs
        num_warmup_steps = int(num_training_steps * 0.1) 

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        print(f"Scheduler initialized. Total steps: {num_training_steps}, Warmup: {num_warmup_steps}")

        # --- Training State 
        best_f1 = 0.0 
        training_history = [] 

        print(f"Starting MODEL 3 training on {device} for {num_train_epochs} epochs...")

        for epoch in range(num_train_epochs):
            start_time = time.time()

            # --- Training Phase ---
            model.train() 
            total_train_loss = 0
            optimizer.zero_grad()

            for i, batch in enumerate(train_loader_m3): # Use M3 loader
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                with autocast(device_type='cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = loss_fct(logits, labels)
                    loss = loss / accumulation_steps 

                scaler_m3.scale(loss).backward() # Use M3 scaler
                total_train_loss += loss.item() * accumulation_steps

                if (i + 1) % accumulation_steps == 0:
                    scaler_m3.step(optimizer)
                    scaler_m3.update()
                    scheduler.step() 
                    optimizer.zero_grad()

                    if (i + 1) % (100 * accumulation_steps) == 0: # Log less often, bigger dataset
                        current_lr = scheduler.get_last_lr()[0]
                        print(
                            f"  M3 S:{strategy_name} G:{gamma_val} | "
                            f"Epoch {epoch + 1}, Batch {i + 1}/{len(train_loader_m3)}, "
                            f"Avg Loss: {total_train_loss / (i + 1):.4f}, LR: {current_lr:.8f}"
                        )

            avg_train_loss = total_train_loss / len(train_loader_m3)

            # --- Validation Phase ---
            model.eval()
            all_logits = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader_m3: # Use M3 loader
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                    all_logits.append(logits.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

            eval_pred = (np.concatenate(all_logits), np.concatenate(all_labels))
            
            # compute_metrics function is already defined from before
            metrics = compute_metrics(eval_pred) 
            epoch_f1 = metrics['f1_weighted']
            epoch_acc = metrics['accuracy']
            epoch_b_acc = metrics['balanced_accuracy']

            end_time = time.time()
            print(f"\n--- MODEL 3 Epoch {epoch + 1}/{num_train_epochs} Complete (Strategy={strategy_name}, Gamma={gamma_val}) ---")
            print(f"Time: {end_time - start_time:.2f}s")
            print(f"Avg Train Loss: {avg_train_loss:.4f}")
            print(f"Validation F1 (Weighted): {epoch_f1:.4f}")
            print(f"Validation Accuracy: {epoch_acc:.4f}")
            print(f"Validation Balanced Accuracy: {epoch_b_acc:.4f}")

            # Save metrics to history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_f1': epoch_f1,
                'val_accuracy': epoch_acc,
                'val_balanced_accuracy': epoch_b_acc
            })

            # Save Best Model *for this run*
            if epoch_f1 > best_f1:
                best_f1 = epoch_f1
                print(f"New best model for M3 {strategy_name}/gamma={gamma_val}! F1: {best_f1:.4f}. Saving to '{best_model_path}'...")
                model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path) # Tokenizer is same, but good practice

            print("----------------------------------\n")

        print(f"--- MODEL 3 Training Complete for Strategy={strategy_name}, Gamma={gamma_val} ---")
        print(f"Best Validation F1 score achieved: {best_f1:.4f}")

        # Add experiment params to this run's history
        for record in training_history:
            record['gamma'] = gamma_val
            record['strategy'] = strategy_name
        
        all_training_histories_m3.extend(training_history)

        # Save results for this experiment 
        all_experiment_results_m3.append({
            'strategy': strategy_name,
            'gamma': gamma_val,
            'best_f1': best_f1,
            'best_model_path': best_model_path
        })
        
        # Track the OVERALL best model 
        if best_f1 > overall_best_f1_m3:
            overall_best_f1_m3 = best_f1
            overall_best_model_path_m3 = best_model_path
            print(f"This is the NEW OVERALL BEST MODEL 3. Path: {overall_best_model_path_m3} !!!")



print("\n==============================================")
print("--- All MODEL 3 Experiments Complete ---")
results_df_m3 = pd.DataFrame(all_experiment_results_m3)
print(results_df_m3.sort_values(by='best_f1', ascending=False))

# Master history df:
all_history_df_m3 = pd.DataFrame(all_training_histories_m3)

print("\n--- Overall Best MODEL 3 ---")
print(f"Path: {overall_best_model_path_m3}")
print(f"Best F1: {overall_best_f1_m3:.4f}")
print("==============================================")


# %% --- Step 5: PLOTTING AND FINAL EVALUATION (MODEL 3) ---

print("\n--- Generating Plots and Final Report for MODEL 3 ---")

# --- Plot 1: Training & Validation Metrics (FOR THE OVERALL BEST M3 RUN) ---
print("Generating plot for the *single best M3* run...")
try:
    best_run_m3 = results_df_m3.loc[results_df_m3['best_f1'].idxmax()]
    best_gamma_val_m3 = best_run_m3['gamma']
    best_strategy_name_m3 = best_run_m3['strategy']
    
    print(f"Best M3 model was: Strategy='{best_strategy_name_m3}', Gamma={best_gamma_val_m3} (F1: {best_run_m3['best_f1']:.4f})")

    best_history_df_m3 = all_history_df_m3[
        (all_history_df_m3['gamma'] == best_gamma_val_m3) & 
        (all_history_df_m3['strategy'] == best_strategy_name_m3)
    ]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(best_history_df_m3['epoch'], best_history_df_m3['train_loss'], 
             label=f'Train Loss (Best M3 Run)', marker='o')
    plt.title(f'Best M3 Run Loss (Strategy: {best_strategy_name_m3}, Gamma: {best_gamma_val_m3})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(best_history_df_m3['epoch'], best_history_df_m3['val_f1'], 
             label=f'Validation F1', marker='o')
    plt.plot(best_history_df_m3['epoch'], best_history_df_m3['val_accuracy'], 
             label=f'Validation Accuracy', marker='s', linestyle='--')
    plt.plot(best_history_df_m3['epoch'], best_history_df_m3['val_balanced_accuracy'], 
             label=f'Validation Balanced Acc', marker='^', linestyle=':')
    plt.title(f'Best M3 Run Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics_BEST_run_m3.png') # New filename
    print("Saved best run training metrics to 'training_metrics_BEST_run_m3.png'")
    plt.close()

except Exception as e:
    print(f"Error generating best run training plot: {e}")


# --- Plot 2: Confusion Matrix & Classification Report (from OVERALL best M3 model on TEST SET) ---
print(f"Loading OVERALL best M3 model from '{overall_best_model_path_m3}' for FINAL TEST evaluation...")
try:
    best_model_m3 = AutoModelForSequenceClassification.from_pretrained(overall_best_model_path_m3)
    best_model_m3.to(device)
    best_model_m3.eval()

    all_logits_test = [] # Rename to avoid confusion
    all_labels_test = [] # Rename to avoid confusion

    with torch.no_grad():
        for batch in test_loader_m3: # <-- USE THE NEW TEST LOADER
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = best_model_m3(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
            all_logits_test.append(logits.cpu().numpy())
            all_labels_test.append(labels.cpu().numpy())

    final_logits_test = np.concatenate(all_logits_test)
    final_labels_test = np.concatenate(all_labels_test)
    final_preds_test = np.argmax(final_logits_test, axis=1)
    
    label_names_m3 = [id2label_m3[i] for i in range(len(id2label_m3))]
    
    print("\n--- Final Classification Report (Best MODEL 3 on **TEST SET**) ---")
    report_m3 = classification_report(final_labels_test, final_preds_test, target_names=label_names_m3)
    print(report_m3)
    
    cm_m3 = confusion_matrix(final_labels_test, final_preds_test)
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm_m3, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names_m3, 
                yticklabels=label_names_m3)
    plt.title('Confusion Matrix (Best Model 3 on **TEST SET**)', fontsize=16) # Updated title
    # ... (rest of plot) ...
    plt.savefig('confusion_matrix_m3_TEST.png') # New filename
    print("Saved confusion matrix plot to 'confusion_matrix_m3_TEST.png'")
    plt.close()

    print("Generating Normalized Confusion Matrix for M3 (on **TEST SET**)...")
    cm_normalized_m3 = confusion_matrix(final_labels_test, final_preds_test, normalize='true')
    
    # ... (rest of heatmap code) ...
    plt.title('Normalized Confusion Matrix M3 (Best Model on **TEST SET**)', fontsize=16) # Updated title
    # ... (rest of plot) ...
    plt.savefig('confusion_matrix_normalized_m3_TEST.png') # New filename
    print("Saved normalized confusion matrix plot to 'confusion_matrix_normalized_m3_TEST.png'")
    plt.close()

except Exception as e:
    print(f"Error during final M3 test evaluation: {e}")


# --- Plot 3: Training Dynamics (Strategy & Gamma Comparison for M3) ---
print("Generating M3 Training Dynamics Comparison Plot (F1)...")
try:
    g = sns.relplot(
        data=all_history_df_m3, # Use M3 history
        x='epoch',
        y='val_f1',
        hue='gamma',
        style='gamma',
        col='strategy',
        kind='line',
        markers=True,
        palette='viridis',
        lw=2,
        height=5,
        aspect=1.2
    )
    g.fig.suptitle('MODEL 3 Validation F1-Score Dynamics by Strategy and Gamma', y=1.03, fontsize=16)
    g.set_axis_labels("Epoch", "Validation F1-Score (Weighted)")
    plt.tight_layout()
    plt.savefig('strategy_gamma_comparison_f1_m3.png') # New filename
    print("Saved plot to 'strategy_gamma_comparison_f1_m3.png'")
    plt.close()

except Exception as e:
    print(f"Error generating M3 dynamics comparison plot: {e}")

print("Generating M3 Training Dynamics Comparison Plot (Balanced Accuracy)...")
try:
    g = sns.relplot(
        data=all_history_df_m3, # Use M3 history
        x='epoch',
        y='val_balanced_accuracy',
        hue='gamma',
        style='gamma',
        col='strategy',
        kind='line',
        markers=True,
        palette='viridis',
        lw=2,
        height=5,
        aspect=1.2
    )
    g.fig.suptitle('MODEL 3 Validation Balanced Accuracy Dynamics by Strategy and Gamma', y=1.03, fontsize=16)
    g.set_axis_labels("Epoch", "Validation Balanced Accuracy")
    plt.tight_layout()
    plt.savefig('strategy_gamma_comparison_balanced_accuracy_m3.png') # New filename
    print("Saved plot to 'strategy_gamma_comparison_balanced_accuracy_m3.png'")
    plt.close()

except Exception as e:
    print(f"Error generating M3 balanced accuracy dynamics plot: {e}")


# --- Plot 4: Precision-Recall Curve (Best M3 Model) ---
print("Generating Precision-Recall Curve for M3...")
try:
    # We need:
    # 1. final_labels_m3 (true integer labels)
    # 2. final_logits_m3 (raw model scores)
    # 3. NUM_LABELS_M3 (total number of classes)
    
    y_scores_m3 = F.softmax(torch.tensor(final_logits_test), dim=1).numpy()
    y_true_bin_m3 = label_binarize(final_labels_test, classes=list(range(NUM_LABELS_M3)))

    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(NUM_LABELS_M3):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true_bin_m3[:, i], y_scores_m3[:, i]
        )
        average_precision[i] = average_precision_score(
            y_true_bin_m3[:, i], y_scores_m3[:, i]
        )

    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin_m3.ravel(), y_scores_m3.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_true_bin_m3, y_scores_m3, average="micro"
    )

    all_recall = np.unique(np.concatenate([recall[i] for i in range(NUM_LABELS_M3)]))
    mean_precision = np.zeros_like(all_recall)
    for i in range(NUM_LABELS_M3):
        mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])
    mean_precision /= NUM_LABELS_M3
    average_precision["macro"] = np.mean(list(average_precision.values()))

    plt.figure(figsize=(10, 8))
    
    plt.plot(
        recall["micro"],
        precision["micro"],
        label=f'Micro-average PR (AP = {average_precision["micro"]:.3f})',
        color='deeppink',
        linestyle=':',
        lw=3
    )
    plt.plot(
        all_recall,
        mean_precision,
        label=f'Macro-average PR (AP = {average_precision["macro"]:.3f})',
        color='navy',
        linestyle='--',
        lw=3
    )

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Multi-Class Precision-Recall Curve (Best Model 3)', fontsize=16)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('precision_recall_curve_m3.png') # New filename
    print("Saved plot to 'precision_recall_curve_m3.png'")
    plt.close()

except Exception as e:
    print(f"Error generating M3 Precision-Recall curve: {e}")

#%% 
