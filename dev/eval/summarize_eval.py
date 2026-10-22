# example run command
# python dev/eval/summarize_eval.py lm-evaluation-harness/results/result774M
# this script is optional, the run_eval.sh should already print these
# but this script can be used to re-print them

استورد json, sys

پایله_2 = sys.argv[1]
چاپ("-"*40)

کلی = {"arc_challenge_25shot.json": "acc_norm",
       "gsm8k_5shot.json": "acc",
       "hellaswag_10shot.json": "acc_norm",
       "mmlu_5shot.json": "acc",
       "truthfulqa_0shot.json": "mc2",
       "winogrande_5shot.json": "acc"
       }

ټول = 0
لكل test في ["arc_challenge_25shot.json", "gsm8k_5shot.json", "hellaswag_10shot.json", "mmlu_5shot.json", "truthfulqa_0shot.json", "winogrande_5shot.json"]:
    معلومات = json.loads(open("./%s/%s"%(پایله_2, test)).اقرأ())
    r_شمېر = 0
    r_ټول = 0
    لكل test_name في معلومات['results']:
      r_شمېر += 1
      r_ټول += معلومات['results'][test_name][کلی[test]]
    score = (r_ټول*100)/r_شمېر
    چاپ(f"{test:<30} : {score:.4f}")
    ټول += score
اوسط = ټول / 6.0
چاپ("-"*40)
چاپ(f"Average Score                  : {average:.4f}")
