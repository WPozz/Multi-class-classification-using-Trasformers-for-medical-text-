
#%%  Libraries: 

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import torch
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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import math
from wordcloud import WordCloud, STOPWORDS
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer


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

def plot_tsne(data, text_col, class_col):
    """
    Plots a t-SNE 2D visualization of the text data,
    colored by class. This is computationally expensive!

        Inputs:
                data = (dataframe) data to analyze
                text_col = (string)  text column to analyze
                class_col = (string) class column to analyze

        Outputs:
                None (just the plot)


    """
    print(f"Generating plot: t-SNE visualization...")
    print("This may take a few minutes...")
    
    # Make a copy to avoid SettingWithCopyWarning
    data = data.copy()
    
    # 1. Create TF-IDF embeddings
    # We limit to 5000 features for performance
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english'
    )
    tfidf_matrix = tfidf.fit_transform(data[text_col])
    
    # 2. Apply t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,  # Standard value
        random_state=42,
        n_iter=300,     # Faster, for exploration
        n_jobs=-1       # Use all cores
    )
    tsne_results = tsne.fit_transform(tfidf_matrix.toarray())
    
    # 3. Add results to DataFrame
    data['tsne-1'] = tsne_results[:, 0]
    data['tsne-2'] = tsne_results[:, 1]
    
    # 4. Plot
    plt.figure(figsize=(16, 12))
    sns.scatterplot(
        x='tsne-1', y='tsne-2',
        hue=class_col,
        palette=sns.color_palette("viridis", n_colors=data[class_col].nunique()),
        data=data,
        legend="full",
        alpha=0.7,
        s=50
    )
    plt.title('t-SNE Visualization of Text Embeddings by Specialty', fontsize=18)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    # Move legend to the side
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('tsne_visualization.png', bbox_inches='tight')
    print("Saved plot to tsne_visualization.png")
    plt.close()



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
    # Run them through a back-translation pipeline (e.g., en-de-en or en-fr-en).  
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

# --- Example of tokenizing a single transcription ---
sample_text = cleaned_data['transcription'].iloc[0]
inputs = tokenizer(sample_text, 
                   return_tensors="pt",  # Return PyTorch tensors
                   max_length=512,       # BERT's maximum
                   truncation=True,      # Truncate long notes
                   padding="max_length") # Pad shorter notes to 512

print("\nTokenized input shape:", inputs['input_ids'].shape)
inputs = inputs.to(device) # Move data to the GPU






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

print("Calculating class weights...")
# Get the unique labels and their counts from the training set
train_labels = train_df['label'].values
unique_labels = np.unique(train_labels)

# Calculate weights
class_weights = compute_class_weight(
    'balanced', 
    classes=unique_labels, 
    y=train_labels
)

# Convert to a PyTorch tensor and move to the GPU
weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"Computed {len(weights_tensor)} class weights.")

# Create the Loss function: 
# We will use this to calculate loss manually
loss_fct = torch.nn.CrossEntropyLoss(weight=weights_tensor)

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
# It adds metric logging, final plots, and a classification report.


# --- 1. Define Compute Metrics Function 
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
        "balanced_accuracy": b_acc
    }

# --- 2. Setup DataLoaders ---
# This assumes 'train_dataset' and 'val_dataset' exist from previous cells
print("Creating DataLoaders...")
train_loader = DataLoader(
    train_dataset,
    batch_size=4,  # Your per_device_train_batch_size
    shuffle=True   # Shuffle training data
)
val_loader = DataLoader(
    val_dataset,
    batch_size=8   # Your per_device_eval_batch_size
)

# --- 3. Setup Optimizer, Loss, and FP16 Scaler ---

# We use AdamW, the standard optimizer for BERT
# We can also try different optimizers 
optimizer = optim.AdamW(model.parameters(), lr=3e-5) # 3e-5 is a good default

# Learning rate: 
# 3e-5 is a good default
# As the optimizer, we can see what happens when we change the learning rate 


# We don't need a loss function, because the model calculates it 
# automatically when we pass 'labels' (outputs.loss)

# --------------------- Question: Since the dataset is unbalanced, 
#                       shouldn't we use a weighted loss function as we said? 
#-----------------------

scaler = GradScaler(device='cuda')


# --- 4. Training Configuration ---

# Let's use a learning rate scheduler 
# Why? 

# A scheduler (specifically one with a "warmup" period) helps stabilize training.
# It starts the learning rate very low, gradually "warms it up" to 
# the target 3e-5, and then slowly decays it over the rest of training. 
# This often leads to better and more stable convergence. 
#
#
#

num_train_epochs = 3 # This can be changed but look at performances 
output_dir = "./clinical_bert_classifier"
best_model_path = f"{output_dir}/best_model"
accumulation_steps = 4 # The gradient_accumulation_steps

# Scheduler Setup 

# Calculate total training steps
num_training_steps = (len(train_loader) // accumulation_steps) * num_train_epochs
num_warmup_steps = int(num_training_steps * 0.1) # 10% warmup is a good default

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
print(f"Scheduler initialized. Total training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")


best_f1 = 0.0  # To track the best model
training_history = [] # List to store metrics 

print(f"Starting training on {device} for {num_train_epochs} epochs...")

for epoch in range(num_train_epochs):
    start_time = time.time()
    
    # --- Training Phase ---
    model.train()  # Set model to training mode
    total_train_loss = 0
    optimizer.zero_grad() 
    
    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Use autocast for mixed precision
        with autocast(device_type='cuda'): 
            # The weighted-loss logic:
            # 1. DO NOT pass labels to the model (data leakage)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # 2. Get the logits
            logits = outputs.logits
    
            # 3. Calculate loss manually with a weighted function
            loss = loss_fct(logits, labels) 
            loss = loss / accumulation_steps # Normalize loss for accumulation

        # Backward pass
        scaler.scale(loss).backward()
        total_train_loss += loss.item() * accumulation_steps
        
        # Optimizer step (every accumulation_steps)
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step() # Step the learning rate scheduler
            optimizer.zero_grad()
            
            # Logging
            if (i + 1) % (50 * accumulation_steps) == 0:
                # Log the current learning rate 
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Avg Loss: {total_train_loss / (i+1):.4f}, LR: {current_lr:.8f}")

    avg_train_loss = total_train_loss / len(train_loader)

# Validation Phase:

# We use F1 score to validate the model but we should add also the "balanced accuracy" metric along with F1. 

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
    print(f"\n--- Epoch {epoch+1}/{num_train_epochs} Complete ---")
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
    
    # Save Best Model
    if epoch_f1 > best_f1:
        best_f1 = epoch_f1
        print(f"New best model found! F1: {best_f1:.4f}. Saving to '{best_model_path}'...")
        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
    
    print("----------------------------------\n")

print("--- Training Complete ---")
print(f"Best Validation F1 score achieved: {best_f1:.4f}")


#%%

# -----------------------------------------------------------------
# 5. PLOTTING AND FINAL EVALUATION  
# -----------------------------------------------------------------



print("\n--- Generating Plots and Final Report ---")

# --- Plot 1: Training & Validation Metrics ---
try:
    history_df = pd.DataFrame(training_history)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Training Loss', marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot F1 and Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history_df['epoch'], history_df['val_f1'], label='Validation F1', marker='o')
    plt.plot(history_df['epoch'], history_df['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Validation Metrics per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print("Saved training metrics plot to 'training_metrics.png'")
    plt.close()

except Exception as e:
    print(f"Error generating training plot: {e}")


# --- Plot 2: Confusion Matrix & Classification Report ---
print(f"Loading best model from '{best_model_path}' for final evaluation...")
try:
    # Load the best model we saved during training
    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
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
    
    # Get the string names for the labels
    # We use the 'id2label' dictionary we created earlier
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

except Exception as e:
    print(f"Error during final evaluation: {e}")

print("\n--- Project Complete ---")

# Other things to add: 

# 1. Normalized Confusion Matrix
# 2. Training Dynamics Plot (Multi-Metric):
# 3. Precision-Recall Curve (Macro-Average)




# Comments on results: 

# The model is currently "bias overfitting"
# These large, generic classes ("Surgery," "Consult," "SOAP/Chart") likely contain so much 
# varied language that they overlap with everything. The model can't find a unique signal, 
# so it guesses, and its guesses are all over the place.

# How to fix this? 

# Focal loss is a thing? 
# Differential Learning rates ? 
# Just surrender on model 2 and try data agumentation (model 3)

#%%  
