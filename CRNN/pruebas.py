import os
from utils import krn_tokenizer
import time

def token_test():
    
    path = '../GT'

    for l in os.listdir(path):
        print(l)
        print(krn_tokenizer(os.path.join(path, l))) 
        time.sleep(1)

if __name__ == '__main__':

    token_test()
