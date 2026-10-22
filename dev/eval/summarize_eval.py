# example run command
# python dev/eval/summarize_eval.py lm-evaluation-harness/results/result774M
# this script is optional, the run_eval.sh should already print these
# but this script can be used to re-print them

আমদানি json, sys

ফলাফল_2 = sys.argv[1]
print("-"*40)

চাবি = {"arc_challenge_25shot.json": "acc_norm",
       "gsm8k_5shot.json": "acc",
       "hellaswag_10shot.json": "acc_norm",
       "mmlu_5shot.json": "acc",
       "truthfulqa_0shot.json": "mc2",
       "winogrande_5shot.json": "acc"
       }

যোগ = 0
জন্য test মধ্যে ["arc_challenge_25shot.json", "gsm8k_5shot.json", "hellaswag_10shot.json", "mmlu_5shot.json", "truthfulqa_0shot.json", "winogrande_5shot.json"]:
    তথ্য = json.loads(open("./%s/%s"%(ফলাফল_2, test)).পঠ())
    r_গুনতি = 0
    r_যোগ = 0
    জন্য test_name মধ্যে তথ্য['results']:
      r_গুনতি += 1
      r_যোগ += তথ্য['results'][test_name][চাবি[test]]
    score = (r_যোগ*100)/r_গুনতি
    print(f"{test:<30} : {score:.4f}")
    যোগ += score
average = যোগ / 6.0
print("-"*40)
print(f"Average Score                  : {average:.4f}")
