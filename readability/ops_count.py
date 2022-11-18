import sys
import argparse
from tqdm import tqdm 
import numpy as np
import math
from collections import Counter

token = {'2': '<TWO>', '3': '<THREE>' , '4': '<FOUR>', '5': '<FIVE>', '6' : '<SIX>',
     '7': '<SEVEN>', '8':'<EIGHT>', '9' : '<NINE>', '10': '<TEN>', '11': '<ELEVEN>', '12' : '<TWELVE>'}
inv_map = {v: int(k) for k, v in token.items()}

def read_file(filename):
    data = []
    with open(filename) as f:
        for line in f:
            data.append(line.strip())
    return data

def parse_ops_file(filename, gold_grade_file=None, by="diff"):
    if gold_grade_file is not None:
        codes = Counter()
        with open(gold_grade_file) as f:
            grades_data = [x.split("\t") for x in read_file(gold_grade_file)]
            src_grades, tgt_grades = zip(*grades_data)


    ops_data = read_file(filename)
    
    num_repos = []
    num_del = []
    num_ins = []
    for line in ops_data:
        r, d, i = map(int, line.strip()[1:-1].split(", "))
        num_repos.append(int(r))
        num_del.append(int(d))
        num_ins.append(int(i))
        if gold_grade_file is not None:
            sg = inv_map[src_grades[i]]
            tg = inv_map[tgt_grades[i]]
            if by=="source":
                grade_diff = sg
            elif by=="target":
                grade_diff = tg
            elif by=="diff":
                grade_diff = sg-tg

            if grade_diff in codes: 
                codes[grade_diff].append([r, d, i])
            else:
                codes[grade_diff] = [[r, d, i]]

    if gold_grade_file is not None:
        avg_codes = {}
        for k in codes:
            avg_codes[k] = np.mean(np.array(codes[k]), axis=0)
            print("Grade %d, Repos: %f, Del: %f, Ins: %f" % (k, np.mean(avg_codes[k][0]), np.mean(avg_codes[k][1]), np.mean(avg_codes[k][2])))


    print("Repos: %f, Del: %f, Ins: %f" % (np.mean(num_repos), np.mean(num_del), np.mean(num_ins)))


def main():
    arg_parser = argparse.ArgumentParser(description='Compute ARI adjacency accuracy')
    arg_parser.add_argument('--ops_file', type=str, default=None)
    # Grades tab separated for source and target
    arg_parser.add_argument('--grade_file', type=str, default=None)
    arg_parser.add_argument('--by', type=str, default="diff")
    
    args = arg_parser.parse_args()
    parse_ops_file(args.ops_file, args.grade_file, args.by)

if __name__ == '__main__':
    main() 