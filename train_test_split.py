import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

class Split:
    def __init__(self, img_folder, csv_filepath) -> None:
        self.img_folder = img_folder
        self.df = pd.read_csv(csv_filepath)
        self.df['filename'] = self.df['filename'].apply(lambda x: x + '.png')
        self.names, self.labels = self.df['filename'], self.df['sign']
        self.names_train, self.names_test, self.labels_train, self.labels_test = train_test_split(self.names, self.labels, test_size = 0.33, random_state = 42)
        self.train_set = pd.concat([self.names_train, self.labels_train], axis=1)
        self.test_set = pd.concat([self.names_test, self.labels_test], axis=1)

    def run(self) -> None:
        Path('train').mkdir(parents=True, exist_ok=True)
        Path('test').mkdir(parents=True, exist_ok=True)
        self.train_set.to_csv('train/train.csv', index=False)
        self.test_set.to_csv('test/test.csv', index=False)
        for name in self.names_train:
            shutil.copy2(self.img_folder + name, 'train/')
        
        for name in self.names_test:
            shutil.copy2(self.img_folder + name, 'test/')