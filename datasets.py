import numpy as np
import os


def _load_bibtex(dir_path):
    """
    Downloads the dataset to ``dir_path``.
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.expanduser(dir_path)

    import urllib.request
    urllib.request.urlretrieve('http://sourceforge.net/projects/mulan/files/datasets/bibtex.rar',os.path.join(dir_path,'bibtex.rar'))

    start_class_id = 1836
    cmd ='unrar e ' + os.path.join(dir_path,'bibtex.rar') + ' ' + dir_path
    os.system(cmd)

    def arff_to_libsvm(lines):
        libsvm_lines = []
        i = 0
        while lines[i].strip() != '@data':
            i+=1
        i+=1
        for line in lines[i:]:
            line = line.strip()[1:-1] # remove starting '{' and ending '}'
            tokens = line.split(',')
            inputs = []
            targets = []
            for tok in tokens:
                id,val = tok.split(' ')
                if int(id) < start_class_id:
                    inputs += [str(int(id)+1) + ':' + val]
                else:
                    targets +=  [ str(int(id)-start_class_id) ]
            libsvm_lines += [','.join(targets) + ' ' + ' '.join(inputs) + '\n']
        return libsvm_lines

    f = open(os.path.join(dir_path,'bibtex-train.arff'))
    train_valid_lines = arff_to_libsvm(f.readlines())
    f.close()

    f = open(os.path.join(dir_path,'bibtex-test.arff'))
    test_lines = arff_to_libsvm(f.readlines())
    f.close()

    import random
    random.seed(12345)
    random.shuffle(train_valid_lines)
    random.shuffle(test_lines)

    valid_size = int(round(0.2*len(train_valid_lines)))
    train_size = len(train_valid_lines)-valid_size
    train_lines = train_valid_lines[:train_size]
    valid_lines = train_valid_lines[train_size:]
    return train_lines, valid_lines, test_lines

def convert_bibtex(train):
  targets = np.zeros(len(train), dtype = int)
  features = np.zeros([len(train), 1836], dtype=int)

  for i, t in enumerate(train):
    h, *t = t.split(' ')
    targets[i] = int(h.split(',')[0])
    idx = [int(i.split(':')[0])-1 for i in t]
    if min(idx) < -1:
      print("uh oh, we have a problem")
      return -1
    features[i, idx] = 1
  return features, targets


def load_bibtex(dir_name):
    train, valid, test = _load_bibtex("bibtex_data")
    x_train, y_train = convert_bibtex(train)
    x_valid, y_valid = convert_bibtex(valid)
    x_test, y_test = convert_bibtex(test)

    x_train = np.vstack([x_train, x_valid])
    y_train = np.concatenate([y_train, y_valid])

    ulab = np.unique(np.concatenate([y_train, y_test]))
    conv = dict(zip(ulab, np.arange(len(ulab))))
    y_train = np.array([conv[i] for i in y_train])
    y_test = np.array([conv[i] for i in y_test])

    return (x_train, y_train), (x_test, y_test)


def convert_eurlex(train):
  targets = np.zeros(len(train), dtype = int)
  features = np.zeros([len(train), 5000], dtype=int)

  for i, t in enumerate(train):
    h, *t = t.split(' ')
    h_val = h.split(',')[0]
    targets[i] = int(h_val) if h_val != '' else -1
    idx = [int(i.split(':')[0])-1 for i in t]
    if min(idx) < -1:
      print("uh oh, we have a problem")
      return -1
    features[i, idx] = 1
  return features, targets

def load_eurlex(dir_name):

    import urllib
    url = "https://www.dropbox.com/s/50szcq78gw8t8rh/eurlex_test.txt?dl=1"
    urllib.request.urlretrieve(url, "eurlex_test.txt")
    url = "https://www.dropbox.com/s/72y1zi2ycwj29g1/eurlex_train.txt?dl=1"
    urllib.request.urlretrieve(url, "eurlex_train.txt")

    f = open("eurlex_train.txt")
    lines = f.readlines()
    x_train, y_train = convert_bibtex(lines[1:])
    idx = np.where(y_train != -1)
    x_train = x_train[idx]
    y_train = y_train[idx]
    f.close()

    f = open("eurlex_test.txt")
    lines = f.readlines()
    x_test, y_test = convert_bibtex(lines[1:])
    idx = np.where(y_test != -1)
    x_test = x_test[idx]
    y_test = y_test[idx]
    f.close()

    ulab = np.unique(np.concatenate([y_train, y_test]))
    conv = dict(zip(ulab, np.arange(len(ulab))))
    y_train = np.array([conv[i] for i in y_train])
    y_test = np.array([conv[i] for i in y_test])

    return (x_train, y_train), (x_test, y_test)
