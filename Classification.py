
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.amp.autocast_mode import autocast 
from torch.amp.grad_scaler import GradScaler
import time
from sklearn.utils.class_weight import compute_class_weight


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

   
def plot_distribution(raw_data, column, title, filename):
    """Helper function to create and save a bar plot of a column's distribution."""
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



    
# We plot the 40 medical specialty classes: 
plot_distribution(raw_data, 'medical_specialty', 'Medical Specialty Distribution', 'medical_specialty_distribution.png')



# Some Notes: 

# There are some classes with very few samples (ex. Hospice or Allergy)
# For now, let's keep the labels as they are and not 

# We have two very different problems, and we need to separate them:
# Problem A: Small, Real Classes (e.t., Allergy / Immunology with 7 samples, Hospice with 6).
# Problem B: Vague, Noisy Classes (e.g., Consult - History and Phy. with 516 samples, SOAP / Chart with 166).

# Problem B can be seen as "garbage in, garbage out problem"


# I thinks the core idea is to build a first model with dropped labels (around -20) and then a second model specifically trained to deal with the biggest class "surgery". 
# Then idk if we can merge the two models into a third one (best of both worlds model) or just testing the two models. 

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
    plot_distribution(cleaned_data, 'medical_specialty', 
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




#%% Model Training
#%% Model Training 

#%% Model Training (Manual PyTorch Loop)  Branch. Trainin + 
# This cell replaces the old 'Trainer' and 'TrainingArguments' cell
#

# --- 1. Define Compute Metrics Function ---
def compute_metrics(eval_pred):
    """
    Computes accuracy and F1 score from logits and labels.
    'eval_pred' is a tuple (logits, labels).
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1) # Get the index of the max logit
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1_weighted": f1,
    }

# --- 2. Setup DataLoaders ---
# We use the train_dataset and val_dataset you already created
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

optimizer = optim.AdamW(model.parameters(), lr=3e-5) 

# Learning rate: 
# 3e-5 is a good default
# As the optimizer, we can see what happens when we change the learning rate 


# We don't need a loss function, because the model calculates it 
# automatically when we pass 'labels' (outputs.loss)

# --------------------- Question: Since the dataset is unbalanced, 
#                       shouldn't we use a weighted loss function as we said? 
#-----------------------


# Initialize the GradScaler for fp16 (mixed precision)
# Why does it says " Il costruttore per la classe "GradScaler" è deprecato
# torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead "

scaler = GradScaler(device = 'cuda')

# --- 4. Training Configuration ---
num_train_epochs = 3        # We could adjust this (see how model changes with more epochs)
output_dir = "./clinical_bert_classifier"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_f1 = 0.0  # To track the best model

print(f"Starting training on {device} for {num_train_epochs} epochs...")

for epoch in range(num_train_epochs):
    start_time = time.time()
    
    # --- Training Phase ---
    model.train()  # Set model to training mode
    total_train_loss = 0
    
    # We apply gradient accumulation manually
    accumulation_steps = 4 # Your gradient_accumulation_steps
    optimizer.zero_grad() # Zero gradients *once* per accumulation cycle
    
    for i, batch in enumerate(train_loader):
        # Move batch to GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Mixed precision forward pass
        with autocast(device_type='cuda', dtype=torch.float16):
            # Get model outputs. The model returns loss
            # because we provided the 'labels'
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Normalize loss for gradient accumulation
            loss = loss / accumulation_steps
        
        # Backward pass (scaled for fp16)
        scaler.scale(loss).backward()
        
        total_train_loss += loss.item() * accumulation_steps # Un-scale for logging
        
        # Optimizer step (only every 'accumulation_steps' batches)
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)  # Unscales gradients and steps
            scaler.update()         # Updates the scale for next pass
            optimizer.zero_grad()   # Zero gradients for the *next* cycle
            
            if (i + 1) % (50 * accumulation_steps) == 0:
                print(f"  Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Avg Loss: {total_train_loss / (i+1):.4f}")

    avg_train_loss = total_train_loss / len(train_loader)

    # --- Validation Phase ---
    model.eval() # Set model to evaluation mode
    all_logits = []
    all_labels = []

    with torch.no_grad(): # Disable gradient calculations
        for batch in val_loader:
            # Move batch to GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass (no labels this time, to get logits)
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits

            # Collect logits and labels from all batches
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Combine results from all validation batches
    eval_pred = (np.concatenate(all_logits), np.concatenate(all_labels))
    
    # Calculate metrics
    metrics = compute_metrics(eval_pred)
    epoch_f1 = metrics['f1_weighted']
    epoch_acc = metrics['accuracy']
    
    end_time = time.time()
    print(f"\n--- Epoch {epoch+1}/{num_train_epochs} Complete ---")
    print(f"Time: {end_time - start_time:.2f}s")
    print(f"Avg Train Loss: {avg_train_loss:.4f}")
    print(f"Validation F1 (Weighted): {epoch_f1:.4f}")
    print(f"Validation Accuracy: {epoch_acc:.4f}")
    
    # --- Save Best Model ---
    if epoch_f1 > best_f1:
        best_f1 = epoch_f1
        print(f"New best model found! F1: {best_f1:.4f}. Saving to '{output_dir}/best_model'...")
        model.save_pretrained(f"{output_dir}/best_model")
        tokenizer.save_pretrained(f"{output_dir}/best_model")
    
    print("----------------------------------\n")

print("--- Project Complete ---")
print(f"Best F1 score achieved: {best_f1:.4f}")


#%% Visualization (Branch 0)

#%% Model Visualization (Run this *after* training)
# This cell loads your saved 'best_model' and generates the final reports.
# It does NOT re-train the model.

# --- Imports (some are redundant, but good for a standalone cell) ---
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast # For fp16
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

print("--- Starting Final Model Evaluation & Visualization ---")

# --- Define variables (in case they were lost) ---
output_dir = "./clinical_bert_classifier"
best_model_path = f"{output_dir}/best_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-create val_loader just in case
# (Assumes 'val_dataset' is still in memory!)
try:
    val_loader = DataLoader(
        val_dataset,
        batch_size=8   # Your per_device_eval_batch_size
    )
    print(f"Validation DataLoader re-created. Using device: {device}")
except NameError:
    print("Error: 'val_dataset' not found. Please re-run the 'Instantiate Datasets' cell.")
    # Stop execution if this fails
    raise
except Exception as e:
    print(f"An error occurred creating DataLoader: {e}")
    raise

# --- Load Model and Run Evaluation ---
print(f"Loading best model from '{best_model_path}'...")
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
            
            # Use autocast for fp16, even during inference, for speed
            with autocast():
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
    # (Assumes 'id2label' is still in memory!)
    label_names = [id2label[i] for i in range(len(id2label))]
    
    # 1. Print Classification Report
    print("\n--- Final Classification Report (Best Model) ---")
    report = classification_report(final_labels, final_preds, target_names=label_names)
    print(report)
    
    # 2. Generate and Save Confusion Matrix
    print("Generating confusion matrix...")
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

except FileNotFoundError:
    print(f"ERROR: Model not found at '{best_model_path}'.")
    print("Please ensure you have successfully trained the model first.")
except NameError as e:
    print(f"ERROR: A required variable was not found in memory (e.g., 'val_dataset' or 'id2label').")
    print(f"Details: {e}")
    print("Please re-run your data preprocessing cells from the top of the notebook.")
except Exception as e:
    print(f"An error during final evaluation: {e}")

print("\n--- Visualization Complete ---")
