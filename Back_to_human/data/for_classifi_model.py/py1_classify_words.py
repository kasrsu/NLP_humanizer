import pandas as pd
import numpy as np
import re
import math
import nltk
from collections import Counter
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import os
# NEW imports for accurate saving and metadata
import json
from datetime import datetime
from joblib import dump, load
from sklearn.metrics import f1_score
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

# Download necessary NLTK data
print("🔄 Downloading NLTK data...")
nltk.download('cmudict', quiet=True)
nltk.download('punkt', quiet=True)
print("✅ NLTK data ready")

# Load CMU Pronouncing Dictionary for syllable counting
try:
    from nltk.corpus import cmudict
    d = cmudict.dict()
except:
    d = {}
    print("⚠️ CMU dictionary not available, syllable count will be estimated")

def count_syllables(word):
    """
    Count syllables in a word using CMU dictionary or fallback method
    
    Args:
        word: Input word
    
    Returns:
        Number of syllables
    """
    word_lower = word.lower()
    
    # Try CMU dictionary first
    if word_lower in d:
        return len([phone for phone in d[word_lower][0] if phone[-1].isdigit()])
    
    # Fallback method: count vowel groups
    word = word_lower
    count = 0
    vowels = 'aeiouy'
    previous_was_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            count += 1
        previous_was_vowel = is_vowel
    
    # Handle silent 'e'
    if word.endswith('e') and count > 1:
        count -= 1
    
    return max(1, count)  # Every word has at least 1 syllable

def calculate_zipf_score(frequency, total_words):
    """
    Calculate Zipf score for a word
    
    Args:
        frequency: Word frequency count
        total_words: Total number of words in corpus
    
    Returns:
        Zipf score (log10 of frequency per billion words)
    """
    if frequency == 0:
        return 0
    
    # Normalize to frequency per billion words
    freq_per_billion = (frequency / total_words) * 1_000_000_000
    return math.log10(freq_per_billion)

def extract_char_ngrams(word, n=3):
    """
    Extract character n-grams from a word
    
    Args:
        word: Input word
        n: N-gram size
    
    Returns:
        List of character n-grams
    """
    word = f"<{word}>"  # Add boundary markers
    return [word[i:i+n] for i in range(len(word)-n+1)]

def create_ngram_features(words, max_features=1000):
    """
    Create character n-gram features using TF-IDF
    
    Args:
        words: List of words
        max_features: Maximum number of features
    
    Returns:
        TF-IDF vectorizer and feature matrix
    """
    print("🔄 Creating character n-gram features...")
    
    # Create character-level analyzer
    def char_ngrams(text):
        return extract_char_ngrams(text, 3)
    
    vectorizer = TfidfVectorizer(
        analyzer=char_ngrams,
        max_features=max_features,
        ngram_range=(1, 1)  # We handle n-grams in char_ngrams function
    )
    
    # Fit and transform
    X_ngrams = vectorizer.fit_transform(words)
    
    print(f"✅ Created {X_ngrams.shape[1]} character n-gram features")
    return vectorizer, X_ngrams

def prepare_word_dataset(word_freq_dict, common_threshold=0.8, uncommon_threshold=0.2):
    """
    Prepare labeled dataset from word frequency dictionary
    
    Args:
        word_freq_dict: Dictionary of {word: frequency}
        common_threshold: Percentile threshold for common words (0.8 = top 20%)
        uncommon_threshold: Percentile threshold for uncommon words (0.2 = bottom 20%)
    
    Returns:
        DataFrame with features and labels
    """
    print("🔄 Preparing word dataset...")
    
    # Convert to list of tuples and sort by frequency
    word_freq_list = [(word, freq) for word, freq in word_freq_dict.items()]
    word_freq_list.sort(key=lambda x: x[1], reverse=True)
    
    total_words = sum(word_freq_dict.values())
    total_unique_words = len(word_freq_list)
    
    print(f"📊 Total words: {total_words:,}")
    print(f"📊 Unique words: {total_unique_words:,}")
    
    # Calculate thresholds
    common_cutoff = int(total_unique_words * (1 - common_threshold))
    uncommon_cutoff = int(total_unique_words * uncommon_threshold)
    
    print(f"🎯 Common words: top {total_unique_words - common_cutoff:,} words")
    print(f"🎯 Uncommon words: bottom {uncommon_cutoff:,} words")
    
    # Prepare data
    data = []
    
    print("🔄 Processing words and extracting features...")
    for i, (word, frequency) in enumerate(tqdm(word_freq_list, desc="Processing words")):
        # Determine label
        if i < common_cutoff:
            label = 'common'
        elif i >= total_unique_words - uncommon_cutoff:
            label = 'uncommon'
        else:
            continue  # Skip middle words for clearer separation
        
        # Extract features
        word_length = len(word)
        syllable_count = count_syllables(word)
        frequency_rank = i + 1
        zipf_score = calculate_zipf_score(frequency, total_words)
        
        # Additional linguistic features
        vowel_count = sum(1 for char in word.lower() if char in 'aeiou')
        consonant_count = word_length - vowel_count
        vowel_ratio = vowel_count / word_length if word_length > 0 else 0
        
        # Complexity features
        unique_chars = len(set(word.lower()))
        char_diversity = unique_chars / word_length if word_length > 0 else 0
        
        # Pattern features
        has_double_letters = bool(re.search(r'(.)\1', word))
        starts_with_vowel = word[0].lower() in 'aeiou' if word else False
        ends_with_vowel = word[-1].lower() in 'aeiou' if word else False
        
        data.append({
            'word': word,
            'frequency': frequency,
            'word_length': word_length,
            'syllable_count': syllable_count,
            'frequency_rank': frequency_rank,
            'zipf_score': zipf_score,
            'vowel_count': vowel_count,
            'consonant_count': consonant_count,
            'vowel_ratio': vowel_ratio,
            'unique_chars': unique_chars,
            'char_diversity': char_diversity,
            'has_double_letters': has_double_letters,
            'starts_with_vowel': starts_with_vowel,
            'ends_with_vowel': ends_with_vowel,
            'label': label
        })
    
    df = pd.DataFrame(data)
    print(f"✅ Created dataset with {len(df):,} labeled words")
    print(f"📊 Label distribution:")
    print(df['label'].value_counts())
    
    return df

def add_ngram_features(df, max_features=500):
    """
    Add character n-gram features to the dataset
    
    Args:
        df: DataFrame with word data
        max_features: Maximum number of n-gram features
    
    Returns:
        Updated DataFrame with n-gram features, vectorizer
    """
    print("🔄 Adding character n-gram features...")
    
    # Create n-gram features
    vectorizer, X_ngrams = create_ngram_features(df['word'].tolist(), max_features)
    
    # Convert sparse matrix to DataFrame
    ngram_feature_names = [f"ngram_{i}" for i in range(X_ngrams.shape[1])]
    ngram_df = pd.DataFrame(X_ngrams.toarray(), columns=ngram_feature_names, index=df.index)
    
    # Combine with original DataFrame
    df_with_ngrams = pd.concat([df, ngram_df], axis=1)
    
    print(f"✅ Added {len(ngram_feature_names)} n-gram features")
    return df_with_ngrams, vectorizer

def train_classification_model(df, test_size=0.2, random_state=42):
    """
    Train classification models to distinguish common vs uncommon words
    
    Args:
        df: DataFrame with features and labels
        test_size: Proportion of data for testing
        random_state: Random seed
    
    Returns:
        Trained models and evaluation results
    """
    print("🔄 Training classification models...")
    
    # Prepare features and labels
    feature_columns = [col for col in df.columns if col not in ['word', 'label']]
    X = df[feature_columns]
    y = df['label']
    
    print(f"📊 Features: {len(feature_columns)}")
    print(f"📊 Samples: {len(X)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"🔄 Training {model_name}...")
        
        if model_name == 'Random Forest':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        accuracy = model.score(X_test_scaled if model_name == 'Logistic Regression' else X_test, y_test)
        # NEW: macro-F1 for better model selection
        f1_macro = f1_score(y_test, y_pred, average='macro')
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'predictions': y_pred,
            'report': report
        }
        
        print(f"✅ {model_name} accuracy: {accuracy:.4f} | F1-macro: {f1_macro:.4f}")
    
    # NEW: also return train split to persist datasets
    return results, scaler, X_train, X_test, y_train, y_test, feature_columns

def visualize_results(df, results, output_dir, y_test):
    """
    Create visualizations of the dataset and model results
    
    Args:
        df: DataFrame with word data
        results: Model results dictionary
        output_dir: Directory to save plots
        y_test: True test labels
    """
    print("🔄 Creating visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature distributions by label
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    numeric_features = ['word_length', 'syllable_count', 'zipf_score', 'vowel_ratio', 'char_diversity', 'frequency_rank']
    
    for i, feature in enumerate(numeric_features):
        if i < len(axes):
            for label in df['label'].unique():
                data = df[df['label'] == label][feature]
                axes[i].hist(data, alpha=0.7, label=label, bins=30)
            axes[i].set_title(f'{feature} Distribution')
            axes[i].legend()
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion matrices - Fixed to use actual y_test
    fig, axes = plt.subplots(1, len(results), figsize=(12, 5))
    if len(results) == 1:
        axes = [axes]
    
    for i, (model_name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['predictions'])  # Fixed: use y_test instead of predictions
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.4f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to {output_dir}")

def save_models_and_data(
    df,
    results,
    scaler,
    vectorizer,
    feature_columns,
    output_dir,
    X_train_idx,
    X_test_idx,
    args
):
    """Save trained models and data in ONNX format for better compatibility."""
    print("🔄 Saving models and data...")
    
    # Create run directory
    run_dir = Path(output_dir) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    for subdir in ["models", "reports", "artifacts", "datasets"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Save datasets
    df.to_csv(run_dir / 'datasets/dataset_full.csv', index=False)
    df_train = df.loc[X_train_idx]
    df_test = df.loc[X_test_idx]
    df_train.to_csv(run_dir / 'datasets/dataset_train.csv', index=False)
    df_test.to_csv(run_dir / 'datasets/dataset_test.csv', index=False)
    
    # Find best model - Fixed comparison logic
    best_model_name = None
    best_entry = None
    best_score = -1.0  # Track single float value instead of tuple
    
    for name, entry in results.items():
        # Combine f1 and accuracy into single score, weighted toward f1
        f1 = entry.get('f1_macro', 0.0)
        accuracy = entry.get('accuracy', 0.0)
        score = (0.7 * f1) + (0.3 * accuracy)  # Weight f1 more heavily
        
        if score > best_score:
            best_score = score
            best_model_name = name
            best_entry = entry
    
    # Save models in ONNX format
    n_features = len(feature_columns)
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    print("🔄 Converting models to ONNX format...")
    for model_name, entry in results.items():
        model = entry['model']
        
        try:
            # Convert to ONNX
            onx = convert_sklearn(model, initial_types=initial_type)
            
            # Save ONNX model
            onnx_path = run_dir / f"models/{model_name.lower().replace(' ', '_')}.onnx"
            with open(onnx_path, "wb") as f:
                f.write(onx.SerializeToString())
            print(f"✅ Saved ONNX model to {onnx_path}")
            
            # Verify ONNX model
            sess = rt.InferenceSession(str(onnx_path))
            input_name = sess.get_inputs()[0].name
            test_input = np.random.rand(1, n_features).astype(np.float32)
            pred_onx = sess.run(None, {input_name: test_input})[0]
            print(f"✅ Verified ONNX model for {model_name}")
            
            # Save metadata
            metadata = {
                "model_name": model_name,
                "accuracy": entry['accuracy'],
                "f1_macro": entry.get('f1_macro'),
                "feature_columns": feature_columns,
                "input_name": input_name,
                "sklearn_version": sklearn.__version__,
                "onnx_version": rt.__version__,
                "saved_at": datetime.now().isoformat(),
                "preprocessing": {
                    "common_threshold": args.common_threshold,
                    "uncommon_threshold": args.uncommon_threshold,
                    "max_ngram_features": args.max_ngram_features,
                    "no_ngrams": args.no_ngrams
                }
            }
            
            with open(run_dir / f"models/{model_name.lower().replace(' ', '_')}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            print(f"⚠️ Failed to convert {model_name} to ONNX: {e}")
            continue
    
    # Save preprocessing components
    preprocessing = {
        "scaler": scaler,
        "vectorizer": vectorizer,
        "feature_columns": feature_columns
    }
    dump(preprocessing, run_dir / "models/preprocessing.joblib")
    
    # Create inference helper script
    inference_script = run_dir / "models/inference.py"
    with open(inference_script, 'w') as f:
        f.write('''
import onnxruntime as rt
import numpy as np
from joblib import load
from pathlib import Path

class ONNXWordClassifier:
    def __init__(self, model_path, preprocessing_path):
        # Load ONNX model
        self.session = rt.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name
        
        # Load preprocessing
        preproc = load(preprocessing_path)
        self.scaler = preproc["scaler"]
        self.vectorizer = preproc["vectorizer"]
        self.feature_columns = preproc["feature_columns"]
    
    def predict(self, features):
        # Scale features
        X = self.scaler.transform(features)
        # Run inference
        return self.session.run(None, {self.input_name: X.astype(np.float32)})[0]

# Example usage:
# classifier = ONNXWordClassifier("best_model.onnx", "preprocessing.joblib")
# prediction = classifier.predict(features)
''')
    
    print(f"\n✨ Models and data saved with ONNX format in: {run_dir}")
    print("📝 Benefits of ONNX format:")
    print("   • Faster inference with ONNX Runtime")
    print("   • Cross-platform compatibility")
    print("   • Language agnostic (use from Python, C++, Java, etc.)")
    print("\n💡 To use the models:")
    print("1. Load the ONNX model with onnxruntime")
    print("2. Use preprocessing.joblib for feature preparation")
    print("3. See inference.py for example code")
    
    return run_dir

def load_word_frequencies(input_path):
    """
    Load word frequencies from various file formats
    
    Args:
        input_path: Path to input file
    
    Returns:
        Dictionary of word frequencies
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"🔄 Loading word frequencies from {input_path}")
    
    if input_path.suffix == '.csv':
        df = pd.read_csv(input_path)
        if 'Word' in df.columns and 'Frequency' in df.columns:
            word_freq = dict(zip(df['Word'], df['Frequency']))
        else:
            raise ValueError("CSV must have 'Word' and 'Frequency' columns")
    
    elif input_path.suffix == '.txt':
        word_freq = {}
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    if '\t' in line:
                        word, freq = line.split('\t', 1)
                        word_freq[word] = int(freq)
                    else:
                        # Assume it's just a list of words, count frequencies
                        word_freq[line] = word_freq.get(line, 0) + 1
    
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    print(f"✅ Loaded {len(word_freq):,} unique words")
    return word_freq

def detect_kaggle_environment():
    """
    Detect if running in Kaggle environment and find available data files
    
    Returns:
        tuple: (is_kaggle, available_files)
    """
    kaggle_paths = ['/kaggle/working', '/kaggle/input']
    available_files = []
    
    for path in kaggle_paths:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(('.txt', '.csv')):
                        available_files.append(os.path.join(root, file))
    
    return len(available_files) > 0, available_files

def find_default_input_file():
    """
    Find default input file in Kaggle environment
    
    Returns:
        str: Path to default input file or None
    """
    # Priority order for default files
    priority_files = [
        '/kaggle/working/cleaned_words.txt',
        '/kaggle/input/cleaned_words.txt',
        '/kaggle/working/words.txt',
        '/kaggle/input/words.txt'
    ]
    
    for file_path in priority_files:
        if os.path.exists(file_path):
            return file_path
    
    # If no priority files found, find any text file
    is_kaggle, available_files = detect_kaggle_environment()
    if is_kaggle and available_files:
        # Prefer .txt files over .csv
        txt_files = [f for f in available_files if f.endswith('.txt')]
        if txt_files:
            return txt_files[0]
        return available_files[0]
    
    return None

def run_kaggle():
    """
    Simple function to run in Kaggle environment without arguments
    """
    print("🎮 Running in Kaggle mode with default settings...")
    
    # Override sys.argv to run without arguments
    import sys
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]  # Keep only script name
    
    try:
        main()
    finally:
        sys.argv = original_argv

def parse_args_safe(parser, is_kaggle: bool):
    """
    Parse args safely in Kaggle/Jupyter by ignoring unknown notebook args (e.g., -f kernel.json).
    """
    import sys
    if is_kaggle:
        args, unknown = parser.parse_known_args()
        if unknown:
            print(f"ℹ️ Ignoring notebook arguments: {' '.join(unknown)}")
        return args
    return parser.parse_args()

def main():
    # Check if we're in Kaggle environment
    is_kaggle, available_files = detect_kaggle_environment()
    default_input = find_default_input_file()
    
    # Auto-detect if running in Jupyter/Kaggle without command line args
    import sys
    if is_kaggle and len(sys.argv) == 1:
        print(f"🐍 Kaggle environment detected!")
        print(f"📁 Available files: {len(available_files)}")
        for f in available_files[:5]:  # Show first 5 files
            print(f"   • {f}")
        if len(available_files) > 5:
            print(f"   ... and {len(available_files) - 5} more files")
        
        if default_input:
            print(f"🎯 Using default input: {default_input}")
        else:
            print("❌ No suitable input files found!")
            print("💡 Please ensure you have a text file with words in /kaggle/working/")
            return 1
    
    parser = argparse.ArgumentParser(description='Word Classification Dataset Creation and Model Training')
    
    if is_kaggle and default_input:
        # Make input optional in Kaggle environment
        parser.add_argument('--input', '-i', default=default_input, 
                          help=f'Input file with word frequencies (default: {default_input})')
    else:
        # Require input in local environment
        parser.add_argument('--input', '-i', required=True, help='Input file with word frequencies')
    
    parser.add_argument('--output', '-o', default='./word_classification_output', help='Output directory')
    parser.add_argument('--common-threshold', type=float, default=0.8, help='Threshold for common words (default: 0.8)')
    parser.add_argument('--uncommon-threshold', type=float, default=0.2, help='Threshold for uncommon words (default: 0.2)')
    parser.add_argument('--max-ngram-features', type=int, default=500, help='Maximum n-gram features (default: 500)')
    parser.add_argument('--no-ngrams', action='store_true', help='Skip n-gram features')
    parser.add_argument('--predict-word', type=str, help='Predict label for a new word')

    # Handle the case when no arguments are provided in Kaggle
    if is_kaggle and len(sys.argv) == 1 and default_input:
        # No arguments provided in Kaggle, use defaults
        args = parser.parse_args([])
    else:
        args = parse_args_safe(parser, is_kaggle)
    
    print("🚀 Starting Word Classification Application")
    print("=" * 60)
    
    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        if is_kaggle:
            print("📁 Available files in Kaggle environment:")
            for f in available_files:
                print(f"   • {f}")
        return 1
    
    try:
        # Load word frequencies
        word_freq_dict = load_word_frequencies(args.input)
        
        # Prepare dataset
        df = prepare_word_dataset(
            word_freq_dict, 
            common_threshold=args.common_threshold,
            uncommon_threshold=args.uncommon_threshold
        )
        
        # Add n-gram features if requested
        vectorizer = None
        if not args.no_ngrams:
            df, vectorizer = add_ngram_features(df, max_features=args.max_ngram_features)
        
        # Train models
        results, scaler, X_train, X_test, y_train, y_test, feature_columns = train_classification_model(df)

        # NEW: create timestamped run directory and pass it through
        run_dir = Path(args.output) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create visualizations into run_dir
        visualize_results(df, results, run_dir, y_test)

        # Save everything (models, bundles, datasets, metadata)
        save_models_and_data(
            df=df,
            results=results,
            scaler=scaler,
            vectorizer=vectorizer,
            feature_columns=feature_columns,
            output_dir=run_dir,
            X_train_idx=X_train.index,
            X_test_idx=X_test.index,
            args=args
        )

        # Find best model for prediction
        best_model_name = None
        best_entry = None
        best_score = -1.0
        for name, entry in results.items():
            f1 = entry.get('f1_macro', 0.0)
            accuracy = entry.get('accuracy', 0.0)
            score = (0.7 * f1) + (0.3 * accuracy)
            if score > best_score:
                best_score = score
                best_model_name = name
                best_entry = entry
        best_model = best_entry['model']

        # If --predict-word is provided, run prediction
        if getattr(args, 'predict_word', None):
            word = args.predict_word
            label, prob = predict_word_label(
                word, scaler, vectorizer, feature_columns, best_model, args
            )
            print(f"\n🔮 Prediction for '{word}':")
            print(f"   • Label: {label}")
            if prob is not None:
                print(f"   • Probabilities: {prob[0]}")
        
        print("\n" + "=" * 60)
        print("✨ WORD CLASSIFICATION COMPLETE!")
        print("=" * 60)
        
        print(f"📊 Dataset: {len(df):,} labeled words")
        print(f"📁 Output directory: {args.output}")
        print("\n🎯 Model Performance:")
        for model_name, result in results.items():
            print(f"   • {model_name}: {result['accuracy']:.4f} accuracy")
        
        # Display sample predictions for verification
        print("\n🔍 Sample Predictions:")
        sample_words = df.sample(10)[['word', 'label', 'zipf_score', 'word_length']].copy()
        for _, row in sample_words.iterrows():
            print(f"   • '{row['word']}' → {row['label']} (zipf: {row['zipf_score']:.2f}, len: {row['word_length']})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

def extract_features_for_word(word):
    """
    Extract features for a single word (same as in dataset)
    """
    word_length = len(word)
    syllable_count = count_syllables(word)
    frequency_rank = 0  # Unknown for new word
    zipf_score = 0      # Unknown for new word
    vowel_count = sum(1 for char in word.lower() if char in 'aeiou')
    consonant_count = word_length - vowel_count
    vowel_ratio = vowel_count / word_length if word_length > 0 else 0
    unique_chars = len(set(word.lower()))
    char_diversity = unique_chars / word_length if word_length > 0 else 0
    has_double_letters = bool(re.search(r'(.)\1', word))
    starts_with_vowel = word[0].lower() in 'aeiou' if word else False
    ends_with_vowel = word[-1].lower() in 'aeiou' if word else False

    features = {
        'word_length': word_length,
        'syllable_count': syllable_count,
        'frequency_rank': frequency_rank,
        'zipf_score': zipf_score,
        'vowel_count': vowel_count,
        'consonant_count': consonant_count,
        'vowel_ratio': vowel_ratio,
        'unique_chars': unique_chars,
        'char_diversity': char_diversity,
        'has_double_letters': has_double_letters,
        'starts_with_vowel': starts_with_vowel,
        'ends_with_vowel': ends_with_vowel
    }
    return features

def predict_word_label(word, scaler, vectorizer, feature_columns, best_model, args):
    """
    Predict label for a new word using trained model
    """
    features = extract_features_for_word(word)
    feature_values = [features[col] for col in feature_columns if col in features]

    # If n-gram features are used, add them
    if not args.no_ngrams and vectorizer is not None:
        ngram_vec = vectorizer.transform([word]).toarray()[0]
        feature_values += list(ngram_vec)

    # Reshape for model input
    X_new = np.array(feature_values).reshape(1, -1)

    # Scale if needed (Logistic Regression uses scaled)
    if hasattr(best_model, 'predict_proba'):
        X_new_scaled = scaler.transform(X_new)
        pred = best_model.predict(X_new_scaled)
        prob = best_model.predict_proba(X_new_scaled)
    else:
        pred = best_model.predict(X_new)
        prob = None

    return pred[0], prob

if __name__ == "__main__":
    main()