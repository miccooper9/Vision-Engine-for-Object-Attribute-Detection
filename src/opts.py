import argparse


def parse_opt():

    parser = argparse.ArgumentParser()


    # train paths
    parser.add_argument('--train_images', type=str, default="./train_images")
    parser.add_argument('--format', type=str, default="jpg")
    parser.add_argument('--train_annotations', type=str, default="./annotations.xml")
    
    #test path
    parser.add_argument('--test_images', type=str, default="./test_images")
    parser.add_argument('--object_preds', type=str, default="./frcnnoutput.csv")

    
    #output path
    parser.add_argument('--output', type=str, default="./output_final.txt")
    
    

    args = parser.parse_args()

    return args
