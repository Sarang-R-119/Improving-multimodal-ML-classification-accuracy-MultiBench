from datasets.imdb.get_data import get_dataloader

folder = '/scratch/eecs545w26_class_root/eecs545w26_class/sarangr/mmimdb_parent/'

# Downloads data into separate loaders for training, validity and testing
train_loader, valid_loader, test_loader = get_dataloader(f"{folder}/multimodal_imdb.hdf5", f"{folder}/mmimdb/mmimdb", num_workers=4, skip_process=True, vgg=False, batch_size=800)

# inspecting the shape of each modality
for batch in train_loader:
    images, text, labels= batch

    print("Image shape: ", images.shape)
    print("Text shape:", text.shape)
    print("Labels shape:", labels.shape)

    break