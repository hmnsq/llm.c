# example run command
# python dev/eval/summarize_eval.py lm-evaluation-harness/results/result774M
# this script is optional, the run_eval.sh should already print these
# but this script can be used to re-print them

దిగుమతి json, sys

ఫలితం_2 = sys.argv[1]
ముద్రించు("-"*40)

కీలకం = {"arc_challenge_25shot.json": "acc_norm",
       "gsm8k_5shot.json": "acc",
       "hellaswag_10shot.json": "acc_norm",
       "mmlu_5shot.json": "acc",
       "truthfulqa_0shot.json": "mc2",
       "winogrande_5shot.json": "acc"
       }

మొత్తం = 0
కొరకు test లో ["arc_challenge_25shot.json", "gsm8k_5shot.json", "hellaswag_10shot.json", "mmlu_5shot.json", "truthfulqa_0shot.json", "winogrande_5shot.json"]:
    దత్తాంశం = json.loads(open("./%s/%s"%(ఫలితం_2, test)).పఠ())
    r_లెక్క = 0
    r_మొత్తం = 0
    కొరకు test_name లో దత్తాంశం['results']:
      r_లెక్క += 1
      r_మొత్తం += దత్తాంశం['results'][test_name][కీలకం[test]]
    score = (r_మొత్తం*100)/r_లెక్క
    ముద్రించు(f"{test:<30} : {score:.4f}")
    మొత్తం += score
సరాసరి = మొత్తం / 6.0
ముద్రించు("-"*40)
ముద్రించు(f"Average Score                  : {average:.4f}")
