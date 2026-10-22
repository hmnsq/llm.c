# example run command
# python dev/eval/summarize_eval.py lm-evaluation-harness/results/result774M
# this script is optional, the run_eval.sh should already print these
# but this script can be used to re-print them

استورد json, sys

نتيجو_2 = sys.argv[1]
ڇاپيو("-"*40)

ڪنجي = {"arc_challenge_25shot.json": "acc_norm",
       "gsm8k_5shot.json": "acc",
       "hellaswag_10shot.json": "acc_norm",
       "mmlu_5shot.json": "acc",
       "truthfulqa_0shot.json": "mc2",
       "winogrande_5shot.json": "acc"
       }

ڪل = 0
لكل test في ["arc_challenge_25shot.json", "gsm8k_5shot.json", "hellaswag_10shot.json", "mmlu_5shot.json", "truthfulqa_0shot.json", "winogrande_5shot.json"]:
    معلومات = json.loads(open("./%s/%s"%(نتيجو_2, test)).اقرأ())
    r_ڳڻپ = 0
    r_ڪل = 0
    لكل test_name في معلومات['results']:
      r_ڳڻپ += 1
      r_ڪل += معلومات['results'][test_name][ڪنجي[test]]
    score = (r_ڪل*100)/r_ڳڻپ
    ڇاپيو(f"{test:<30} : {score:.4f}")
    ڪل += score
اوسط = ڪل / 6.0
ڇاپيو("-"*40)
ڇاپيو(f"Average Score                  : {average:.4f}")
