# noisy-danger

![](https://mars-images.imgix.net/seobot/dataheadhunters.com/65d3fe4cbb7642d3cbdb5203-abc2fc3fac35580a6b7b95994eaf764c.png?auto=compress)

__Image Credit to [Handling Noisy Data: Smoothing and Filtering Techniques](https://dataheadhunters.com/academy/handling-noisy-data-smoothing-and-filtering-techniques/)__

## From Noise to Knowledge

Text analysis is a powerful tool, but noisy text – riddled with typos, slang, and inconsistencies – can throw a wrench in the works. This is especially true for Natural Language Processing (NLP) tasks like sentiment analysis or topic modeling. 

Moreover, the complexity increases when malicious actors exploit these weaknesses through sophisticated techniques.

### Prompt Injection Attacks

**[Prompt injection](https://github.com/ranfysvalle02/prompt-injection-demo/)** is a method where attackers manipulate the input given to an LLM to produce unintended outputs. This technique involves disguising malicious instructions as benign user inputs, effectively tricking the model into executing harmful commands. For instance, if a user inputs a prompt that appears harmless but is crafted to override the model's intended behavior, the LLM may execute actions that it was designed to avoid.

- **Example of Prompt Injection**: An attacker might input a command that instructs the LLM to ignore previous instructions and perform an action that could compromise security or privacy.

The risk of prompt injection is particularly high in applications that concatenate trusted prompts with untrusted user inputs. This vulnerability allows attackers to manipulate how the model interprets and executes commands, leading to potential data breaches or misuse of sensitive functionalities.

### Jailbreaking Techniques

**Jailbreaking** refers to tactics used to bypass the safety filters embedded within LLMs. Unlike prompt injections, which focus on manipulating input, jailbreaking aims to convince the model that it can disregard its built-in safeguards entirely. This can be achieved through various methods:

- **Human-Written Jailbreaks**: These are crafted prompts designed to make the LLM believe it has permission to act outside its constraints.
- **Automated Jailbreaking Scripts**: These scripts attempt to brute-force the model into producing desired outputs by appending random characters or instructions.

For example, a well-known jailbreak prompt known as **DAN (Do Anything Now)** creates an alter ego for the model, allowing it to bypass its alignment and restrictions. The effectiveness of such techniques highlights the ongoing arms race between developers implementing safeguards and attackers seeking ways to circumvent them.

### Mitigation Strategies

To combat these threats, developers are exploring various strategies:

- **Output Filtering**: Instead of solely focusing on preventing malicious prompts from reaching the model, some suggest filtering outputs for suspicious content post-processing. This approach allows for greater flexibility while maintaining security.
- **Immutable System Prompts**: Creating prompts that cannot be altered by user inputs could help ensure that models adhere strictly to their intended functions without being influenced by external commands.

### Cleaning Up the Mess: Text Preprocessing

Noisy text is like a messy room – it needs organization before you can find anything. The first step is text cleaning, which involves techniques like:

* **Lowercasing:** Converting all characters to lowercase for consistency.
* **Punctuation Removal:** Removing irrelevant symbols like commas and periods.
* **Stopword Removal:** Eliminating common words like "the" and "a" that don't carry much meaning.
* **Tokenization:** Breaking down the text into meaningful units like words or phrases.

Python's Natural Language Toolkit (NLTK) library provides tools for these tasks. Here's an example of cleaning a noisy sentence:

```python
# Before cleaning: "PyTorch Lightning is GREAT!! NLP is AMAZING!!!! #NLP #PyTorch @User"
# After cleaning: ["pytorch", "lightning", "great", "nlp", "amazing", "nlp", "pytorch", "user"]
```


Here’s an example of cleaning a noisy text using NLTK:

```python
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords if not available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercasing
    text = text.lower()
    
    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization and removing stopwords
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens

# Example noisy text
text = "PyTorch Lightning is GREAT!! NLP is AMAZING!!!! #NLP #PyTorch @User"
cleaned_text = clean_text(text)
print(cleaned_text) # ['pytorch', 'lightning', 'great', 'nlp', 'amazing', 'nlp', 'pytorch', 'user']
```

### From Text to Numbers: Text Vectorization

Clean text is great, but deep learning models need numerical data to work their magic. This is where text vectorization comes in. Techniques like [Bag-of-Words (BoW)](https://github.com/ranfysvalle02/just-a-bag-of-words), and word embeddings convert text into numerical representations.

### Understanding Text Vectorization Techniques

Text vectorization is the process of converting text data into numerical representations that can be understood by machine learning models. Here are some common techniques:

* **Bag-of-Words (BoW):** This is a simple technique that represents each document as a bag (multiset) of its words, disregarding grammar and word order but keeping track of frequency. For example, the sentence "The cat sat on the mat" might be represented as `{"the": 2, "cat": 1, "sat": 1, "on": 1, "mat": 1}`.

* **Word Embeddings:** This is a type of word representation that allows words with similar meanings to have similar representations. Techniques like Word2Vec can generate these embeddings, which capture semantic relationships between words.

Here's an example of how to create a BoW representation using Python's `CountVectorizer`:

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['The cat sat on the mat.', 'The dog sat on the log.', 'Cats and dogs are great.']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())

"""
['and' 'are' 'cat' 'cats' 'dog' 'dogs' 'great' 'log' 'mat' 'on' 'sat'
 'the']
[[0 0 1 0 0 0 0 0 1 1 1 2]
 [0 0 0 0 1 0 0 1 0 1 1 2]
 [1 1 0 1 0 1 1 0 0 0 0 0]]
"""
```

### Advanced Techniques for Handling Noisy Text Data

While the techniques above can handle a lot of NLP tasks, there are more advanced techniques that can be used for complex or noisy text data:

* **Transformer Models (BERT, RoBERTa, etc.):** These models use transformer architectures, which rely on [self-attention mechanisms](https://github.com/ranfysvalle02/ai-self-attention/). They have been pre-trained on large corpora and can generate context-aware word embeddings. For noisy text data, these models can capture the context of words and generate more accurate representations.

* **Handling Out-of-Vocabulary Words:** Sometimes, you might encounter words that are not in your vocabulary, especially with noisy text data. Techniques to handle this include using a special "unknown" token to represent all unknown words, or using subword tokenization techniques (like Byte Pair Encoding) that can represent unknown words based on known subwords.

Remember, these advanced techniques might be more computationally intensive and could require more time to train. However, they can often provide better performance, especially on complex or noisy text data.

### Building the Model: Extracting Insights

![](https://nexocode.com/cdn-cgi/image/width=600,quality=75/images/NLP-tasks.webp)

__Image Credit to [Leveraging Natural Language Processing (NLP) for Healthcare and Pharmaceutical Companies](https://nexocode.com/blog/posts/natural-language-processing-healthcare/)__

Once your data is cleaned and vectorized, you can build your NLP model. PyTorch Lightning provides a structured framework to handle the deep learning pipeline, allowing you to focus on the modeling aspect rather than boilerplate code. Common NLP tasks include:

* **Sentiment Analysis**: Classifying text as positive, negative, or neutral.
* **Topic Modeling**: Identifying hidden thematic structures within a collection of documents.
* **Text Classification**: Categorizing text into predefined classes like "sports" or "politics".

Let's walk through a full demo that processes simple synthetic text data and builds an NLP model with word embeddings. This will focus on tokenization, creating word embeddings, and running a classifier to give you a solid understanding of how NLP models work at a fundamental level.

### `demo.py`

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

# 1. Synthetic Data
data = [
    ("hello world", 1),
    ("goodbye world", 0),
    ("hello everyone", 1),
    ("goodbye everyone", 0)
]

# 2. Tokenization - simple split on space
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.word_to_idx = self.build_vocab(data)
    
    def build_vocab(self, data):
        words = set()
        for sentence, _ in data:
            words.update(sentence.split())  # Split each sentence into words
        word_to_idx = {word: idx for idx, word in enumerate(words)}
        return word_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, label = self.data[idx]
        tokenized = [self.word_to_idx[word] for word in sentence.split()]
        return torch.tensor(tokenized, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# 3. Simple Embedding-based Classifier
class TextClassifier(pl.LightningModule):
    def __init__(self, vocab_size, embed_dim=10):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, 1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)  # Average embeddings for simplicity
        return self.fc(embedded).squeeze()

    def training_step(self, batch, batch_idx):
        tokens, labels = batch
        outputs = self(tokens)
        loss = self.loss(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

# 4. DataLoader
dataset = TextDataset(data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 5. Model Training
model = TextClassifier(vocab_size=len(dataset.word_to_idx))
trainer = pl.Trainer(max_epochs=10, log_every_n_steps=1)

print("Training started...")
trainer.fit(model, dataloader)
print("Training completed!")

# 6. Model Testing (on new data)
test_sentence = "hello world"
test_tokens = torch.tensor([dataset.word_to_idx[word] for word in test_sentence.split()], dtype=torch.long)
model.eval()
with torch.no_grad():
    prediction = model(test_tokens.unsqueeze(0))  # Unsqueeze to add batch dimension
    print(f"\nTest Prediction for '{test_sentence}':", torch.sigmoid(prediction).item())

```

### Breakdown of the Demo:

1. **Synthetic Data**: 
   - The dataset consists of simple sentences like `"hello world"` with binary labels (`1` or `0`).

2. **Tokenization**: 
   - Tokenization is done using Python's basic `.split()` to break each sentence into words.
   - A vocabulary is built based on the words in the dataset. Each word is assigned an index.

3. **TextDataset Class**: 
   - A PyTorch `Dataset` is created to handle tokenized data. The `__getitem__` method converts sentences into lists of token indices, which are passed to the model.

4. **TextClassifier**: 
   - The model uses a simple embedding layer to turn tokens into vectors. The vectors are averaged to create a sentence representation.
   - A fully connected layer (`fc`) classifies the sentence based on the embeddings.
   - We use `BCEWithLogitsLoss` since it’s a binary classification task.

5. **Training**: 
   - We use PyTorch Lightning's `Trainer` to train the model for 10 epochs, logging the training loss.

6. **Prediction**:
   - After training, we test the model with the sentence `"hello world"`, showing the predicted output using `torch.sigmoid` to get a probability.

### Running the Demo
1. Install necessary packages:

   ```bash
   pip install torch pytorch-lightning
   ```

2. Run the demo:

   ```bash
   python demo.py
   ```


#### OUTPUT
```
Training started...

  | Name      | Type              | Params | Mode 
--------------------------------------------------------
0 | embedding | Embedding         | 40     | train
1 | fc        | Linear            | 11     | train
2 | loss      | BCEWithLogitsLoss | 0      | train
--------------------------------------------------------
51        Trainable params
0         Non-trainable params
51        Total params
0.000     Total estimated model params size (MB)
3         Modules in train mode
0         Modules in eval mode
/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:424: The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=9` in the `DataLoader` to improve performance.
Epoch 9: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 299.21it/s, v_num=38]`Trainer.fit` stopped: `max_epochs=10` reached.
Epoch 9: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 202.12it/s, v_num=38]
Training completed!
Test Prediction for 'hello world': 0.6120739579200745
```

### Conquering the Noise: Challenges and Solutions

Noisy text throws some curveballs that need to be addressed:

* **Misspellings**: Pre-trained word embeddings might struggle with misspelled words. Consider incorporating spelling correction in preprocessing.
* **Domain-Specific Language**: Pre-trained models may not handle industry-specific jargon well. Fine-tuning models like BERT on your own data can improve accuracy.
* **Data Augmentation**: Techniques like synonym replacement or back-translation can artificially enlarge your dataset for better handling of noisy text.

### Conclusion

In conclusion, dealing with noisy text data is a complex but surmountable challenge. With the right tools and techniques, we can transform this seeming chaos into valuable insights. From the initial steps of text preprocessing to the advanced techniques of handling out-of-vocabulary words and leveraging transformer models, we have the tools available to transform noise into knowledge.
