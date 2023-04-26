import torch
from datasets import load_dataset
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from transformers import BertTokenizerFast


def data_prepare(dataset_name, mode, tokenizer, max_len):
    # if we are in train mode, we will load two columns (i.e., text and label).
    if mode == 'train':
        # Use pandas to load dataset
        data = load_dataset("glue", dataset_name, split="train")
        # Convert data into torch tensors
        labels = torch.tensor(data["label"])

    # if we are in predict mode, we will load one column (i.e., text).
    elif mode == 'validation':
        data = load_dataset("glue", dataset_name, split="validation")
        # Convert data into torch tensors
        labels = torch.tensor(data["label"])

    # elif mode == 'test':
    #     data = load_dataset("glue", dataset_name, split="test")
    #     # Convert data into torch tensors
    #     labels = torch.tensor(data["label"])[:100]

    else:
        print("the type of mode should be either 'train' or 'predict'. ")
        return

    # Create sentence and label lists
    content = data

    # We need to add a special token at the beginning for BERT to work properly.
    if dataset_name != "cola":
        content = ["[CLS] " + text1 + "[SEP]" + text2 for text1, text2 in zip(content["sentence1"],
                                                                             content["sentence2"])]
    else:
        content = ["[CLS] " + text1 for text1 in content["sentence"]]

    # Import the BERT tokenizer, used to convert our text into tokens that correspond to BERT's vocabulary.
    tokenized_texts = [tokenizer.tokenize(text) for text in content]

    # if the sequence is longer the maximal length, we truncate it to the pre-defined maximal length
    tokenized_texts = [text[:max_len + 1] for text in tokenized_texts]

    # We also need to add a special token at the end.
    tokenized_texts = [text + ['[SEP]'] for text in tokenized_texts]
    print("Tokenize the first sentence:\n", tokenized_texts[0])

    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    print("Index numbers of the first sentence:\n", input_ids[0])

    # Pad our input seqeunce to the fixed length (i.e., max_len) with index of [PAD] token
    pad_ind = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
    input_ids = pad_sequences(input_ids, maxlen=max_len + 2, dtype="long", truncating="post", padding="post",
                              value=pad_ind)
    print("Index numbers of the first sentence after padding:\n", input_ids[0])

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for pad tokens
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # Convert all of our data into torch tensors, the required datatype for our model
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)
    #### REF END ####

    return inputs, labels, masks


def dataloader(batch_size=32, dataset_name="cola", model="bert-base-uncased", max_len=32):
    # tokenizer for pre-trained BERT model
    tokenizer = BertTokenizerFast.from_pretrained(model, do_lower_case=True)

    # preprocess the data
    train_inputs, train_labels, train_masks = data_prepare(dataset_name=dataset_name, mode="train",
                                                           tokenizer=tokenizer, max_len=max_len)
    validation_inputs, validation_labels, validation_masks = data_prepare(dataset_name=dataset_name, mode="validation",
                                                                          tokenizer=tokenizer, max_len=max_len)
    # test_inputs, test_labels, test_masks = data_prepare(dataset_name=dataset_name, mode="test", tokenizer=tokenizer,
    #                                                     max_len=max_len)

    # print(train_inputs.shape, train_labels.shape, train_masks.shape)
    # print(validation_inputs.shape, validation_labels.shape, validation_masks.shape)
    # print(test_inputs.shape, test_labels.shape, test_masks.shape)
    # take training samples in random order in each epoch.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_dataloader = DataLoader(train_data,
                                  sampler=RandomSampler(train_data),  # Select batches randomly
                                  batch_size=batch_size)

    # Read validation set sequentially.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_dataloader = DataLoader(validation_data,
                                       sampler=SequentialSampler(validation_data),  # Pull out batches sequentially.
                                       batch_size=batch_size)

    # test_data = TensorDataset(test_inputs, test_masks, test_labels)
    # test_dataloader = DataLoader(test_data,
    #                              sampler=SequentialSampler(test_data),  # Pull out batches sequentially.
    #                              batch_size=batch_size)

    return train_dataloader, validation_dataloader, tokenizer