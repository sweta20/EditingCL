import sys
from readability import Readability
import argparse
from tqdm import tqdm 
import pickle
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy
import math
from utils import get_words

token = {'2': '<TWO>', '3': '<THREE>' , '4': '<FOUR>', '5': '<FIVE>', '6' : '<SIX>',
     '7': '<SEVEN>', '8':'<EIGHT>', '9' : '<NINE>', '10': '<TEN>', '11': '<ELEVEN>', '12' : '<TWELVE>'}
inv_map = {v: k for k, v in token.items()}

def clip_value(x, min_val=2, max_val=12):
    if(x<min_val):
        return min_val
    elif(x>max_val):
        return max_val
    else:
        return x

def get_text_grade_score(inp_text, clip_val, grade_type):
    rd = Readability(inp_text.strip())
    if grade_type=="ARI":
        grade = rd.ARI()
    elif grade_type=="FLESCH":
        grade = rd.FleschKincaidGradeLevel()
    else:
        print("Select Grade type ARI/FLESCH")
        exit(0)
    if clip_val:
        return clip_value(grade)
    return grade

def compute_pmi(pmi_weights, pred_file, gold_grade_file):
    with open(pred_file) as f:
        pred_data = f.readlines()
    PMI = pickle.load(open(pmi_weights,"rb"))

    with open(gold_grade_file) as f:
        grades = [int(inv_map[x.strip()]) for x in f.readlines()]

    W = 0
    PMI_score = 0.0
    for i in range(len(pred_data)):
        grade=grades[i]
        words = get_words(pred_data[i])
        W+=len(words)
        for word in words:
            if (word, grade) in PMI:
                PMI_score+=max(PMI[(word, grade)], 0)
    print("MPMI: ", PMI_score/W)


def compute_accuracy(ref_file, pred_file, gold_grade_file=None, clip_val=False, grade_type="ARI"):
    with open(ref_file) as f:
        ref_data = f.readlines()
    with open(pred_file) as f:
        pred_data = f.readlines()

    acc = []
    pred_grades = []
    tgt_grades = []
    for i in tqdm(range(len(ref_data))):
        try:
            ari_ref_grade =  get_text_grade_score(ref_data[i], clip_val, grade_type)
            ari_pred_grade =  get_text_grade_score(pred_data[i], clip_val, grade_type)
            pred_grades.append(ari_pred_grade)
            tgt_grades.append(ari_ref_grade)
            acc.append(int(abs(ari_ref_grade-ari_pred_grade) <=1))
        except:
            continue

    if gold_grade_file is None:
        bins = numpy.linspace(2, 12, 11)
        buckets = numpy.digitize(tgt_grades, bins)
    else:
         with open(gold_grade_file) as f:
            buckets = [int(inv_map[x.strip()]) for x in f.readlines()]

    pred_bucket = {}
    acc_bucket = {}
    ref_bucket = {}
    for i in range(len(pred_grades)):
        if buckets[i] in pred_bucket:
            pred_bucket[buckets[i]].append(pred_grades[i])
            acc_bucket[buckets[i]].append(acc[i])
            ref_bucket[buckets[i]].append(tgt_grades[i])
        else:
            pred_bucket[buckets[i]] = [pred_grades[i]]
            acc_bucket[buckets[i]] = [acc[i]]
            ref_bucket[buckets[i]] = [tgt_grades[i]]

    pred_avg_grade = {k: sum(v)/len(v) for k, v in pred_bucket.items()}
    ref_avg_grade = {k: sum(v)/len(v) for k, v in ref_bucket.items()}
    avg_accuracy_grade = {k: sum(v)/len(v) for k, v in acc_bucket.items()}

    score = sum(acc)/len(acc)
    print("Adjcency %s accuracy: " %(grade_type) + str(score))

    corr = numpy.corrcoef(tgt_grades, pred_grades)
    print("Corre %s: " %(grade_type) + str(corr[0][1]))

    mse = 0.0
    # print("Average Grade for each bucket: ")
    for k in pred_bucket:
        # print(token[str(int(k))], ": ", pred_avg_grade[k])
        # print(k, pred_avg_grade[k])
        mse += (k - pred_avg_grade[k]) **2
        # mse += (pred_avg_grade[k] - ref_avg_grade[k]) **2

    mse /= len(pred_bucket)

    print("RMSE: ", math.sqrt(mse))

    # This should be for different grade differences and not for a target grade
    print("Average %s Accuracy for each bucket:  " %(grade_type))
    for k in avg_accuracy_grade:
        print(token[str(int(k))], ": ", avg_accuracy_grade[k])


def main():
    arg_parser = argparse.ArgumentParser(description='Compute ARI adjacency accuracy')
    arg_parser.add_argument('--ref_file', type=str, default=None)
    arg_parser.add_argument('--pred_file', type=str, default=None)
    arg_parser.add_argument('--grade_file', type=str, default=None)
    arg_parser.add_argument('--clip_val', action='store_true', dest='clip_val')
    arg_parser.add_argument('--grade_type', type=str, default="ARI")
    arg_parser.add_argument('--pmi-weights', type=str, default=None)

    args = arg_parser.parse_args()
    compute_accuracy(args.ref_file, args.pred_file, args.grade_file, args.clip_val, args.grade_type)

    # if args.pmi_weights is not None and args.grade_file is not None:
        # compute_pmi(args.pmi_weights, args.pred_file, args.grade_file)



if __name__ == '__main__':
    main() 