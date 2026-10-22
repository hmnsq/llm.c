# example run command
# python dev/eval/summarize_eval.py lm-evaluation-harness/results/result774M
# this script is optional, the run_eval.sh should already print these
# but this script can be used to re-print them

ಆನಯನ json, sys

RESULT = sys.argv[1]
print("-"*40)

ಕೀಲಿ = {"arc_challenge_25shot.json": "acc_norm",
       "gsm8k_5shot.json": "acc",
       "hellaswag_10shot.json": "acc_norm",
       "mmlu_5shot.json": "acc",
       "truthfulqa_0shot.json": "mc2",
       "winogrande_5shot.json": "acc"
       }

total = 0
ಪ್ರತಿ test ಅನ್ತಃ ["arc_challenge_25shot.json", "gsm8k_5shot.json", "hellaswag_10shot.json", "mmlu_5shot.json", "truthfulqa_0shot.json", "winogrande_5shot.json"]:
    data = json.loads(open("./%s/%s"%(RESULT, test)).ಪಠ())
    r_ಎಣಿಕೆ = 0
    r_total = 0
    ಪ್ರತಿ test_name ಅನ್ತಃ data['results']:
      r_ಎಣಿಕೆ += 1
      r_total += data['results'][test_name][ಕೀಲಿ[test]]
    score = (r_total*100)/r_ಎಣಿಕೆ
    print(f"{test:<30} : {score:.4f}")
    total += score
average = total / 6.0
print("-"*40)
print(f"Average Score                  : {average:.4f}")
