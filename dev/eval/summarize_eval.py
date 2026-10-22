# example run command
# python dev/eval/summarize_eval.py lm-evaluation-harness/results/result774M
# this script is optional, the run_eval.sh should already print these
# but this script can be used to re-print them

استورد json, sys

نتیجه_2 = sys.argv[1]
چاپ("-"*40)

کلید = {"arc_challenge_25shot.json": "acc_norm",
       "gsm8k_5shot.json": "acc",
       "hellaswag_10shot.json": "acc_norm",
       "mmlu_5shot.json": "acc",
       "truthfulqa_0shot.json": "mc2",
       "winogrande_5shot.json": "acc"
       }

مجموع = 0
لكل test في ["arc_challenge_25shot.json", "gsm8k_5shot.json", "hellaswag_10shot.json", "mmlu_5shot.json", "truthfulqa_0shot.json", "winogrande_5shot.json"]:
    داده = json.loads(open("./%s/%s"%(نتیجه_2, test)).اقرأ())
    r_شمار = 0
    r_مجموع = 0
    لكل test_name في داده['results']:
      r_شمار += 1
      r_مجموع += داده['results'][test_name][کلید[test]]
    score = (r_مجموع*100)/r_شمار
    چاپ(f"{test:<30} : {score:.4f}")
    مجموع += score
معدل = مجموع / 6.0
چاپ("-"*40)
چاپ(f"Average Score                  : {average:.4f}")
