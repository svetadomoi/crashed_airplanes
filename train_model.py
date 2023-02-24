from train_test_split import Split
from model import ResNetLike
split_done = True

if split_done == False:
    spl = Split('avia-train/', 'avia-train/all.csv')
    spl.run()

model = ResNetLike()
model.build()
model.run()
model.model.load_weights('w.h5')
