# example run command
# python dev/eval/summarize_eval.py lm-evaluation-harness/results/result774M
# this script is optional, the run_eval.sh should already print these
# but this script can be used to re-print them

आयात json, sys

परिणाम_2 = sys.argv[1]
print("-"*40)

कुंजी = {"arc_challenge_25shot.json": "acc_norm",
       "gsm8k_5shot.json": "acc",
       "hellaswag_10shot.json": "acc_norm",
       "mmlu_5shot.json": "acc",
       "truthfulqa_0shot.json": "mc2",
       "winogrande_5shot.json": "acc"
       }

योग = 0
क्रम test in ["arc_challenge_25shot.json", "gsm8k_5shot.json", "hellaswag_10shot.json", "mmlu_5shot.json", "truthfulqa_0shot.json", "winogrande_5shot.json"]:
    तथ्य = json.loads(open("./%s/%s"%(परिणाम_2, test)).म_पढ़ो())
    r_गिनती = 0
    r_योग = 0
    क्रम test_name in तथ्य['results']:
      r_गिनती += 1
      r_योग += तथ्य['results'][test_name][कुंजी[test]]
    score = (r_योग*100)/r_गिनती
    print(f"{test:<30} : {score:.4f}")
    योग += score
average = योग / 6.0
print("-"*40)
print(f"Average Score                  : {average:.4f}")
