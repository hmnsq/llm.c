# example run command
# python dev/eval/summarize_eval.py lm-evaluation-harness/results/result774M
# this script is optional, the run_eval.sh should already print these
# but this script can be used to re-print them

درآمد json, sys

نتیجہ_2 = sys.argv[1]
چھاپو("-"*40)

کلید = {"arc_challenge_25shot.json": "acc_norm",
       "gsm8k_5shot.json": "acc",
       "hellaswag_10shot.json": "acc_norm",
       "mmlu_5shot.json": "acc",
       "truthfulqa_0shot.json": "mc2",
       "winogrande_5shot.json": "acc"
       }

کل = 0
برائے test میں ["arc_challenge_25shot.json", "gsm8k_5shot.json", "hellaswag_10shot.json", "mmlu_5shot.json", "truthfulqa_0shot.json", "winogrande_5shot.json"]:
    ڈیٹا = json.loads(open("./%s/%s"%(نتیجہ_2, test)).اقرأ())
    r_گنتی = 0
    r_کل = 0
    برائے test_name میں ڈیٹا['results']:
      r_گنتی += 1
      r_کل += ڈیٹا['results'][test_name][کلید[test]]
    score = (r_کل*100)/r_گنتی
    چھاپو(f"{test:<30} : {score:.4f}")
    کل += score
اوسط_قدر = کل / 6.0
چھاپو("-"*40)
چھاپو(f"Average Score                  : {average:.4f}")
