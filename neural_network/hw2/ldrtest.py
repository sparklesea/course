import argparse
import os
import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--mode",
                    default="dependent",
                    type=str)

def split_label(raw_label, cls):
    new_label = np.zeros_like(raw_label)
    new_label[raw_label==cls] = 1

    return new_label

def calculate_acc(score, label):
    score = np.array(score)[..., 1]
    prediction = np.argmax(score, axis=0)
    hit = np.zeros_like(prediction)
    hit[prediction==label] = 1
    acc = np.average(hit)
    
    return acc

def main():
    args = parser.parse_args()
    
    data_dir = "./SEED-IV"

    data = []
    for sess in sorted(os.listdir(data_dir)):
        sess_dir = os.path.join(data_dir, sess)
        session = []
        for sample in sorted(os.listdir(sess_dir)):
            sample_dir = os.path.join(sess_dir, sample)
            sample = {}
            for file in sorted(os.listdir(sample_dir)):
                sample[file.split(".")[0]] = np.load(os.path.join(sample_dir, file))

            session.append(sample)
        data.append(session)
    
    if args.mode == "dependent": 
        models = []
        for sess in data:
            models_sess = []
            for sample in sess:
                model = []
                for cls in range(4):
                    clf = SVC(kernel='linear',probability=True)
                    X = sample["train_data"].reshape(sample["train_data"].shape[0], -1)
                    y = split_label(sample["train_label"], cls)
                    clf.fit(X, y)
                    model.append(clf)
                models_sess.append(model)
            models.append(models_sess)

        test_acc = []
        for idx_sess, sess in enumerate(data):
            for idx_sample, sample in enumerate(sess):
                scores_4cls = []
                for cls in range(4):
                    model = models[idx_sess][idx_sample][cls]
                    X = sample["test_data"].reshape(sample["test_data"].shape[0], -1)
                    scores_4cls.append(model.predict_proba(X))

                test_acc.append(calculate_acc(scores_4cls, sample["test_label"]))

    elif args.mode == "independent":
        test_acc = []
        for idx_sample in tqdm(range(len(data[0]))):
            train_data_list = []
            train_label_list = []
            test_data_list = []
            test_label_list = []
            for idx_sess in range(len(data)):
                test_data_list.append(data[idx_sess][idx_sample]["train_data"])
                test_label_list.append(data[idx_sess][idx_sample]["train_label"])
                test_data_list.append(data[idx_sess][idx_sample]["test_data"])
                test_label_list.append(data[idx_sess][idx_sample]["test_label"])

                for idx_train_sample in range(len(data[0])):
                    if idx_train_sample == idx_sample: continue
                    
                    train_data_list.append(data[idx_sess][idx_sample]["train_data"])
                    train_label_list.append(data[idx_sess][idx_sample]["train_label"])
                    train_data_list.append(data[idx_sess][idx_sample]["test_data"])
                    train_label_list.append(data[idx_sess][idx_sample]["test_label"])

            train_data = np.concatenate(train_data_list)
            train_label = np.concatenate(train_label_list)
            test_data = np.concatenate(test_data_list)
            test_label = np.concatenate(test_label_list)

            X = train_data.reshape(train_data.shape[0], -1)
            test_X = test_data.reshape(test_data.shape[0], -1)
            
            scores_4cls = []
            for cls in range(4):
                clf = SVC(probability=True)
                y = split_label(train_label, cls)
                clf.fit(X, y)
                scores_4cls.append(clf.predict_proba(test_X))

            test_acc.append(calculate_acc(scores_4cls, test_label))
        
    else: assert False

    final_acc = np.average(test_acc)
    print("Mode " + args.mode + " accuracy :", final_acc)

if __name__ == "__main__":
    main()